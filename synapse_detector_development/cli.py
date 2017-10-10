"""
ARIADNE Synapse Detector Development.

Tools for synapse detector development for MICrONS Team 1 TA3.

Usage:
    synapse-detector-development create [--metadata-only] [<name> <path>]
    synapse-detector-development upload <model-file> <weights-file> <metadata> <custom-layer-file>
    synapse-detector-development pickle <metadata>
    synapse-detector-development evaluate <model-file> <weights-file> <metadata> [<custom-layer-file>] [<rh-config>]
    synapse-detector-development -h | --help
    synapse-detector-development --version

Options:
    -h --help   Show this screen.
    --version   Show version.

Commands for synapse detector development are:
    create    Initialize a new synapse detector project with example files.
    evaluate  Run a trained classifier through the ARIADNE pipeline.
    upload    Submit a synapse detector model for evaluation.

"""
from synapse_detector_development import commands

from docopt import docopt


MODULE_DOC = __doc__

def main(docstring=MODULE_DOC):
    arguments = docopt(docstring, version='0.0.1')

    print(arguments)
    cmd = commands.get_command_class(**arguments)
    if cmd is not None:
        cmd.run()


if __name__ == '__main__':
    main(__doc__)
