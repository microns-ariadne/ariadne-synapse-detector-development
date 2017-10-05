#!/usr/bin/env python
"""
ARIADNE Synapse Detector Development.

Tools for synapse detector development for MICrONS Team 1 TA3.

Usage:
    synapse-detector-development [--version] [-h | --help] <command> [<args>...]
    synapse-detector-development init [-n STRING] <path>
    synapse-detector-development submit <model-file> <weights-file> <metadata>
    synapse-detector-development -h | --help
    synapse-detector-development --version

Options:
    -h, --help   Show this screen.
    --version    Show version.

Commands for synapse detector development are:
    init    Initialize a new synapse detector project with example files.
    submit  Submit a synapse detector model for evaluation.

Use synapse-detector-development help <command> for more information about a
specific command.

"""
import os
import subprocess
import sys

from docopt import docopt


def main(docstring):
    arguments = docopt(docstring, version='0.0.1')

    commands_dir = os.path.join(os.path.dirname(__file__), 'commands')
    commands = os.listdir(commmands_dir)

    cmdfile = '{}.py'.format(args['<command>'])
    argv = arguments['<args>']
    if len(argv) > 0:
        helpcmd = '{}.py'.format(argv[0])

    if cmdfile in commands:
        args = [os.path.join(commands_dir, cmdfile)] + argv
        sys.exit(subprocess.call(args))
    elif arguments['command'] == 'help' and helpcmd in commands:
        args = [os.path.join(commands_dir, helpcmd), '--help']
        sys.exit(subprocess.call(args))
    elif arguments['<command>'] in ['help', None]:
        exit(call([__file__, '--help']))
    else:
        print('{} is not a valid command.'.format(arguments['<command>']))
        sys.exit(docstring)


if __name__ == '__main__':
    main(__doc__)
