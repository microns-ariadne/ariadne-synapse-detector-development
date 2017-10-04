"""
ARIADNE Synapse Detector Development.

Usage:
    asdd create [-n STRING] <path>
    asdd upload <model-file> <weights-file> <metadata>
    asdd -h | --help
    asdd --version

Options:
    -h --help   Show this screen.
    --version   Show version.
    -n STRING   Name of the new synapse detector. [default: synapse-detector]

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
