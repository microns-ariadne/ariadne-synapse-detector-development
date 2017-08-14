#! /bin/bash
#
# Script to run a synapse classifier on the R0 test dataset
#
# The following environment variables should be defined
#
# MICRONS_TMP_DIR - a directory for intermediate results
# MICRONS_ROOT_DIR - the directory for the report.json file output by the script
# MICRONS_CLASSIFIER - the classifier .pkl file
#
# The install procedure in LAB_INSTRUCTIONS.md should be performed and
# the command, "source activate ariadne_microns_pipeline" should be performed
# to activate the conda environment.
#
########################

if [ ! -d $MICRONS_TMP_DIR ]; then
    mkdir -p $MICRONS_TMP_DIR
fi

if [ ! -d $MICRONS_ROOT_DIR ]; then
    mkdir -p $MICRONS_ROOT_DIR
fi

#######################################################################
#
# Make the classifier pkl file
#
#######################################################################

MICRONS_CLASSIFIER=$MICRONS_ROOT_DIR/syanpse-classifier.pkl

python pickle_synapse_classifier.py $MICRONS_CLASSIFIER

#######################################################################
#
# Read parameters from the classifier file.
#
#######################################################################

MICRONS_PAD_X=`python -c 'import cPickle;print cPickle.load(open("'$MICRONS_CLASSIFIER'")).get_x_pad()'`
MICRONS_PAD_Y=`python -c 'import cPickle;print cPickle.load(open("'$MICRONS_CLASSIFIER'")).get_y_pad()'`
MICRONS_PAD_Z=`python -c 'import cPickle;print cPickle.load(open("'$MICRONS_CLASSIFIER'")).get_z_pad()'`

MICRONS_WIDTH=$(( 1496 - 2*$MICRONS_PAD_X ))
MICRONS_HEIGHT=$(( 1496 - 2*$MICRONS_PAD_Y ))
MICRONS_DEPTH=$(( 145 - 2*$MICRONS_PAD_Z ))

if [ `python -c 'import cPickle;print "synapse" in cPickle.load(open("'$MICRONS_CLASSIFIER'")).get_class_names()'` == "True" ]; then
    MICRONS_WANTS_TRANSMITTER_RECEPTOR=
else
    MICRONS_WANTS_TRANSMITTER_RECEPTOR=--wants-transmitter-receptor-synapse-maps
fi

###################################################################
#
# Set up RH_CONFIG to point to lab-rh-config.yaml
#
###################################################################
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

#----------------------------------------------------------------
#
# Rewrite the .rh-config.yaml file to point at the luigi DB in
#         the temp directory.
#
#---------------------------------------------------------------
MICRONS_LUIGI_DB=sqlite:///$MICRONS_TMP_DIR/luigi-task-hist.db
ORIG_RH_CONFIG_FILENAME=$SCRIPTPATH/lab-rh-config.yaml
export RH_CONFIG_FILENAME=$MICRONS_TMP_DIR/.rh-config.yaml

python -c 'import yaml;cfg=yaml.safe_load(open("'$ORIG_RH_CONFIG_FILENAME'"));cfg["luigid"]["db_connection"]="'$MICRONS_LUIGI_DB'";yaml.dump(cfg,open("'$RH_CONFIG_FILENAME'","w"))'

#----------------------------------------------------
#
# Start ipc workers
#
#----------------------------------------------------

microns-ipc-broker >> $MICRONS_TMP_DIR/ipc-broker.log \
                  2>> $MICRONS_TMP_DIR/ipc-broker.err.log & \
MICRONS_IPC_BROKER_PID=$!
microns-ipc-worker \
     --lifetime=86400 \
     >> $MICRONS_TMP_DIR/ipc-worker.log \
    2>> $MICRONS_TMP_DIR/ipc-worker.err.log &
MICRONS_IPC_WORKER_PID=$!


##################################################################
#
# Start Butterfly on port 2198
#
##################################################################

MICRONS_BUTTERFLY_PORT=2198
bfly $MICRONS_BUTTERFLY_PORT \
     > "$MICRONS_TMP_DIR"/bfly.log \
    2>> "$MICRONS_TMP_DIR"/bfly.err &
BFLY_PROCESS="$!"

#################################################################
#
# Start the luigi daemon
#
################################################################

export LUIGI_CONFIG_PATH="$MICRONS_TMP_DIR"/luigi.cfg
MICRONS_SCHEDULER_PORT=8082
cat <<EOF > "$LUIGI_CONFIG_PATH"
[scheduler]
record_task_history=True
state_path=$MICRONS_TMP_DIR/luigi-state.pickle

[task_history]
db_connection=$MICRONS_LUIGI_DB

[resources]
cpu_count=1
gpu_count=1
memory = 70000000000

[worker]
keep_alive=True


EOF

luigid --logdir="$MICRONS_TMP_DIR" \
       --port $MICRONS_SCHEDULER_PORT \
     >> "$MICRONS_TMP_DIR"/luigid.out \
    2>> "$MICRONS_TMP_DIR"/luigid.err &
LUIGID_PROCESS="$!"

######################################################################
#
# Run Luigi
#
#####################################################################

set -x
luigi --module ariadne_microns_pipeline.pipelines.synapse_score_pipeline \
      --workers=1 \
      --scheduler-port=$MICRONS_SCHEDULER_PORT \
      ariadne_microns_pipeline.SynapseScorePipelineTask \
      --experiment=ECS_iarpa_201610_gt_4x6x6 \
      --sample=neocortex \
      --dataset=sem \
      --channel=raw \
      --synapse-channel=synapse \
      --url=http://localhost:"$MICRONS_BUTTERFLY_PORT"/api \
      --volume='{"x":'$MICRONS_PAD_X',"y":'$MICRONS_PAD_Y',"z":'$MICRONS_PAD_Z',"width":'$MICRONS_WIDTH',"height":'$MICRONS_HEIGHT',"depth":'$MICRONS_DEPTH'}'\
      --temp-dir=$MICRONS_TMP_DIR \
      --root-dir=$MICRONS_ROOT_DIR \
      --pixel-classifier=$MICRONS_CLASSIFIER \
      $MICRONS_WANTS_TRANSMITTER_RECEPTOR \
      --synapse-min-overlap-pct=5 \
      --synapse-min-gt-overlap-pct=25 \
      --synapse-report-path=$MICRONS_ROOT_DIR/report.json \
      >> $MICRONS_TMP_DIR/luigi.log \
      2>> $MICRONS_TMP_DIR/luigi.err.log
set +x

#-------------------------------------------------------------------
#
# Clean up
#
#-------------------------------------------------------------------

kill "$BFLY_PROCESS"
kill "$LUIGID_PROCESS"
kill "$MICRONS_IPC_BROKER_PID"
kill "$MICRONS_IPC_WORKER_PID"


