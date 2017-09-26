"""
ARIADNE Synapse Detector Development.

Usage:
    asdd create [-n STRING] <path>
    asdd upload [(<model-file> <weights-file>)]
    asdd -h | --help
    asdd --version

Options:
    -h --help   Show this screen.
    --version   Show version.
    -n STRING   Name of the new synapse detector. [default: synapse-detector]

"""
import .commands

from docopt import docopt


def main():
    arguments = docopt(__doc__, version='0.0.1')

    cmd = commands.get_command_class(arguments)
    if cmd is not None:
        cmd.run()
