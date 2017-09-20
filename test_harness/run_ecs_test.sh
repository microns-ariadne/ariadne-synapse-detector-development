#! /bin/bash
#SBATCH --gres=gpu:1
#SBATCH -n 6
#SBATCH -t 01:00:00
#SBATCH -p cox
#SBATCH --mem=90000
#SBATCH -o /n/coxfs01/leek/results/2017-05-04_test/slurm-%j.log
#SBATCH -e /n/coxfs01/leek/results/2017-05-04_test/slurm-%j.err
#
# Script to run the Luigi pipeline on the ECS test volume
#
# Environment variables:
#
# MICRONS_RESULT_DIR:
#     the root of the directory tree for final and intermediate results
#
export MICRONS_RESULT_DIR=$1
export MICRONS_PORT=9094
#
# MICRONS_N_GPUS:
#     the # of GPUS reserved
#
export MICRONS_N_GPUS=4
#
# MICRONS_N_WORKERS:
#     the # of workers to run
#
export MICRONS_N_WORKERS=4

export MICRONS_USE_IPC=1
#
# ARIADNE_MICRONS_PIPELINE_PIPELINETASK_PIXEL_CLASSIFIER_PATH:
#     the path to the pixel classifier pickle
#
export MICRONS_PIXEL_CLASSIFIER=$2
#
# ARIADNE_MICRONS_PIPELINE_PIPELINETASK_NEUROPROOF_CLASSIFIER_PATH:
#     the path to the Neuroproof classifier
#
export MICRONS_NEUROPROOF_CLASSIFIER=$3

#
# RH_CONFIG_FILENAME:
#     the path to the .rh-config.yaml file
#
export RH_CONFIG_FILENAME=$4

#-----------------------------------------------------
#
# Ensure that the supplied arguments are properly set up
#
#-----------------------------------------------------

if [ ! -d "$MICRONS_RESULT_DIR" ]; then
    mkdir -v "$MICRONS_RESULT_DIR"
    if [ "$?" != "0" ]; then
        exit 1
    fi
fi

if [ ! -f "$MICRONS_PIXEL_CLASSIFIER" ]; then
    echo "ERROR: Invalid pixel classifier $MICRONS_PIXEL_CLASSIFIER"
    exit 1
fi

if [ ! -f "$MICRONS_NEUROPROOF_CLASSIFIER" ]; then
    echo "ERROR: Invalid pixel classifier $MICRONS_NEUROPROOF_CLASSIFIER"
    exit 1
fi

if [ ! -f "$RH_CONFIG_FILENAME" ]; then
    echo "ERROR: Invalid pixel classifier $RH_CONFIG_FILENAME"
    exit 1
fi

#-----------------------------------------------------
#
# Environment initialization
#
#-----------------------------------------------------

source ~jkinniso/Public/apps/x86_64/scripts/env.sh
module load anaconda
module load cuda/8.0
module load cudnn/5.1
source activate ariadne_microns_pipeline

#-----------------------------------------------------
#
# Start butterfly
#
#-----------------------------------------------------

bfly 2198 -e $RH_CONFIG_FILENAME > "$MICRONS_RESULT_DIR"/bfly.log 2>> "$MICRONS_RESULT_DIR"/bfly.err &
BFLY_PROCESS="$!"

#----------------------------------------------------
#
# Start ipc workers
#
#----------------------------------------------------

if [ "$MICRONS_USE_IPC" ]; then

    microns-ipc-broker >> $MICRONS_RESULT_DIR/ipc-broker.log \
                      2>> $MICRONS_RESULT_DIR/ipc-broker.err.log &
    MICRONS_IPC_BROKER_PID=$!
    for i in $(seq 1 $MICRONS_N_GPUS)
        do
            export CUDA_VISIBLE_DEVICES=$(($i-1))
            THEANO_FLAGS="device=gpu$(($i-1))" MICRONS_IPC_WORKER_GPU=$(($i-1)) strace microns-ipc-worker \
           --lifetime=86400 \
                              >> $MICRONS_RESULT_DIR/ipc-worker"$i".log \
                             2>> $MICRONS_RESULT_DIR/ipc-worker"$i".err.log &
           MICRONS_IPC_WORKER_PIDS[$i]=$!
        done
fi
#----------------------------------------------------
#
# Write a temporary luigi.cfg file
#
#----------------------------------------------------
e
export LUIGI_CONFIG_PATH="$MICRONS_RESULT_DIR"/luigi.cfg

cat <<EOF > "$LUIGI_CONFIG_PATH"
[scheduler]
record_task_history=True
state_path=$MICRONS_RESULT_DIR/luigi-state.picklee

[task_history]
db_connection=sqlite:///$MICRONS_RESULT_DIR/luigi-task-hist.db

[resources]
cpu_count=$MICRONS_N_WORKERS
gpu_count=$MICRONS_N_GPUS
memory = 70000000000

[worker]
keep_alive=True


EOF

#-----------------------------------------------------
#e
# Start luigid
#
#-----------------------------------------------------

luigid --logdir="$MICRONS_RESULT_DIR" \
       --port $MICRONS_PORT \
     >> "$MICRONS_RESULT_DIR"/luigid.out \
    2>> "$MICRONS_RESULT_DIR"/luigid.err &
LUIGID_PROCESS="$!"

#-----------------------------------------------------
#
# Butterfly variables
#
#-----------------------------------------------------

ARIADNE_MICRONS_PIPELINE_PIPELINETASK_EXPERIMENT=ECS_iarpa_201610_gt_4x6x6
ARIADNE_MICRONS_PIPELINE_PIPELINETASK_SAMPLE=neocortex
ARIADNE_MICRONS_PIPELINE_PIPELINETASK_DATASET=sem
ARIADNE_MICRONS_PIPELINE_PIPELINETASK_CHANNEL=raw
ARIADNE_MICRONS_PIPELINE_PIPELINETASK_GT_CHANNEL=gt
ARIADNE_MICRONS_PIPELINE_PIPELINETASK_SYNAPSE_CHANNEL=synapse
ARIADNE_MICRONS_PIPELINE_PIPELINETASK_URL=http://0.0.0.0:2198/api

#-----------------------------------------------------
#
# Volume - Run on the whole of S1, leaving borders for those
#          that need them.
#          There are eight chunks of 125 MVoxels each.
#
#-----------------------------------------------------
MICRONS_TEST_WIDTH=1496
MICRONS_TEST_HEIGHT=1496
MICRONS_TEST_DEPTH=145

MICRONS_PAD_X=`python -c 'import cPickle;print cPickle.load(open("'$MICRONS_PIXEL_CLASSIFIER'")).get_x_pad()'`
MICRONS_PAD_Y=`python -c 'import cPickle;print cPickle.load(open("'$MICRONS_PIXEL_CLASSIFIER'")).get_y_pad()'`
MICRONS_PAD_Z=`python -c 'import cPickle;print cPickle.load(open("'$MICRONS_PIXEL_CLASSIFIER'")).get_z_pad()'`

MICRONS_BLOCK_WIDTH=$(($MICRONS_TEST_WIDTH-$MICRONS_PAD_X*2))
MICRONS_BLOCK_HEIGHT=$(($MICRONS_TEST_HEIGHT-$MICRONS_PAD_Y*2))
MICRONS_BLOCK_DEPTH=$(($MICRONS_TEST_DEPTH-$MICRONS_PAD_Z*2))

#-----------------------------------------------------
#
# Segmentation parameters
#
#-----------------------------------------------------
MICRONS_SIGMA_XY=3.0
MICRONS_SIGMA_Z=0.4
MICRONS_THRESHOLD=1
MICRONS_MINIMUM_DISTANCE_XY=5
MICRONS_MINIMUM_DISTANCE_Z=1.5
MICRONS_NEUROPROOF_THRESHOLD=0.20
#-----------------------------------------------------
#
# Scratch space
#
#-----------------------------------------------------

ARIADNE_MICRONS_PIPELINE_PIPELINETASK_TEMP_DIRS='["'"$MICRONS_RESULT_DIR"'"]'

#-----------------------------------------------------
#
# Reports
#
#-----------------------------------------------------

ARIADNE_MICRONS_PIPELINE_PIPELINETASK_PIPELINE_REPORT_LOCATION="$MICRONS_RESULT_DIR"/timing_report.csv
ARIADNE_MICRONS_PIPELINE_PIPELINETASK_STATISTICS_CSV_PATH="$MICRONS_RESULT_DIR"/statistics_report.csv

#------------------------------------------------------------------
#
# The Luigi logging configuration file
#
#------------------------------------------------------------------

cat <<EOF > $MICRONS_RESULT_DIR/luigi-client.cfg
[core]
no_configure_logging=True
default-scheduler-port=$MICRONS_PORT
[resources]
cpu_count=$MICRONS_N_WORKERS
gpu_count = $MICRONS_N_GPUS
memory = 70000000000
EOF

#-------------------------------------------------------------------
#
# Run Luigi
#
#-------------------------------------------------------------------
set -x

LUIGI_CONFIG_PATH=$MICRONS_RESULT_DIR/luigi-client.cfg \
luigi --workers=$MICRONS_N_WORKERS \
      --scheduler-port $MICRONS_PORT \
      --module ariadne_microns_pipeline.pipelines \
      ariadne_microns_pipeline.PipelineTask \
      --wants-transmitter-receptor-synapse-maps \
      --pixel-classifier-path=$MICRONS_PIXEL_CLASSIFIER \
      --neuroproof-classifier-path=$MICRONS_NEUROPROOF_CLASSIFIER \
      --experiment="$ARIADNE_MICRONS_PIPELINE_PIPELINETASK_EXPERIMENT" \
      --sample="$ARIADNE_MICRONS_PIPELINE_PIPELINETASK_SAMPLE" \
      --dataset="$ARIADNE_MICRONS_PIPELINE_PIPELINETASK_DATASET" \
      --channel="$ARIADNE_MICRONS_PIPELINE_PIPELINETASK_CHANNEL" \
      --gt-channel=$ARIADNE_MICRONS_PIPELINE_PIPELINETASK_GT_CHANNEL \
      --synapse-channel=$ARIADNE_MICRONS_PIPELINE_PIPELINETASK_SYNAPSE_CHANNEL \
      --url="$ARIADNE_MICRONS_PIPELINE_PIPELINETASK_URL" \
      --volume='{"x":'$MICRONS_PAD_X',"y":'$MICRONS_PAD_Y',"z":'$MICRONS_PAD_Z',"width":'$MICRONS_BLOCK_WIDTH',"height":'$MICRONS_BLOCK_HEIGHT',"depth":'$MICRONS_BLOCK_DEPTH'}' \
      --block-width=$MICRONS_BLOCK_WIDTH \
      --block-height=$MICRONS_BLOCK_HEIGHT \
      --block-depth=$MICRONS_BLOCK_DEPTH \
      --classifier-block-width=$MICRONS_BLOCK_WIDTH \
      --classifier-block-height=$MICRONS_BLOCK_HEIGHT \
      --classifier-block-depth=$MICRONS_BLOCK_DEPTH \
      --sigma-xy=$MICRONS_SIGMA_XY \
      --sigma-z=$MICRONS_SIGMA_Z \
      --minimum-distance-xy=$MICRONS_MINIMUM_DISTANCE_XY \
      --minimum-distance-z=$MICRONS_MINIMUM_DISTANCE_Z \
      --threshold=$MICRONS_THRESHOLD \
      --np-x-pad=200 \
      --np-y-pad=200 \
      --np-z-pad=30 \
      --np-threshold=$MICRONS_NEUROPROOF_THRESHOLD \
      --np-cores=2 \
      --neuroproof-version=FAST \
      --min-synapse-neuron-contact=500 \
      --min-synapse-area=1000 \
      --synapse-max-size-2d=15000 \
      --synapse-min-size-2d=250 \
      --min-synapse-depth=4 \
      --synapse-xy-sigma=.5 \
      --synapse-z-sigma=.5 \
      --synapse-threshold=128 \
      --temp-dir=$MICRONS_RESULT_DIR \
      --root-dir=$MICRONS_RESULT_DIR \
      --stitched-segmentation-location="$MICRONS_RESULT_DIR/stitched_segmentation.h5" \
      --pipeline-report-location="$ARIADNE_MICRONS_PIPELINE_PIPELINETASK_PIPELINE_REPORT_LOCATION" \
      --connectivity-graph-location=$MICRONS_RESULT_DIR/connectivity-graph.json \
      --synapse-connection-location=$MICRONS_RESULT_DIR/synapse-connections.json \
      --statistics-csv-path=$MICRONS_RESULT_DIR/segmentation-statistics.csv \
      --synapse-statistics-path=$MICRONS_RESULT_DIR/synapse-statistics.json \
      --index-file-location=$MICRONS_RESULT_DIR/index.json \
      --synapse-min-overlap-pct=10.0 \
      >> "$MICRONS_RESULT_DIR"/luigi.log \
      2>> "$MICRONS_RESULT_DIR"/luigi.err.log

#-------------------------------------------------------------------
#
# Clean up
#
#-------------------------------------------------------------------

kill -9 "$BFLY_PROCESS"
kill "$LUIGID_PROCESS"
kill "$MICRONS_IPC_BROKER_PID"
for MICRONS_IPC_WORKER_PID in "${MICRONS_IPC_WORKER_PIDS[@]}"
do
    kill "$MICRONS_IPC_WORKER_PID"
done

if [ -f "$MICRONS_RESULT_DIR/luigi.err.log" ]; then
    err="$(grep -q ERROR "$MICRONS_RESULT_DIR/luigi.err.log")"
    ops="$(grep -q sqlite3.OperationalError "$MICRONS_RESULT_DIR/luigi.err.log")"
else
    exit 1
fi

if [ "$err" == "0" ] && [ "$ops" != "0" ]; then
    exit 1
fi

exit 0
