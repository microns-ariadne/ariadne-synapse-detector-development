'''Pickle a synapse classifier

This script reads the file, "classifier.yaml", in this directory and
creates a pickle file in a designated location. See example_classifier.yaml
for an example of the classifier file's format.

arguments:
python pickle_synapse_classifier.py <pickle-filename>

'''
import cPickle
import os
import sys
import yaml

from ariadne_microns_pipeline.classifiers.keras_classifier \
     import KerasClassifier, NormalizeMethod

src_file = os.path.join(
    os.path.dirname(sys.argv[0]),
    "classifier.yaml")
if not os.path.isfile(src_file):
    print "There is no classifier parameters file (%s) in the" % src_file
    print "base directory of the ariadne-synapse-detector-development "
    print "repo. Please copy example-classifier.yaml to classifier.yaml "
    print "and fill it out for your classifier."
    raise IOError("%s is missing" % src_file)

dest_file = sys.argv[1]

params = yaml.load(open(src_file))["classifier"]

transpose = map(
    lambda _:None if isinstance(_, basestring) and _.lower() == "none" else _,
    params["transpose"])

k = KerasClassifier(
    model_path=params["model-path"],
    weights_path=params["weights-path"],
    xypad_size=params["xy-pad-size"],
    zpad_size=params["z-pad-size"],
    block_size=params["block-size"],
    normalize_method=NormalizeMethod[params["normalize-method"]],
    downsample_factor=params["downsample-factor"],
    xy_trim_size=params["xy-trim-size"],
    z_trim_size=params["z-trim-size"],
    classes=params["classes"],
    stretch_output=params["stretch-output"],
    invert=params["invert"],
    split_positive_negative=params["split-positive-negative"],
    normalize_offset=params["normalize-offset"],
    normalize_saturation_level=params["normalize-saturation-level"],
    transpose=transpose
)

with open(dest_file, "w") as fd:
    cPickle.dump(k, fd)
