"""Base class for creating subsequent command classes and utility functions.

Classes
-------
BaseCommand

Functions
---------
get_command_class
    Return the class corresponding to the requested command.
"""
from .create import CreateCommand
from .upload import UploadCommand


REGISTERED_COMMANDS = {
    'create': CreateCommand,
    'upload': UploadCommand
}


def get_command_class(**kws):
    for command in REGISTERED_COMMANDS:
        if kws[command]:
            return REGISTERED_COMMANDS[command](**kws)
    return None

class BaseCommand(object):
    """Abstract command object that others should inherit from."""

    def run():
        return None
