"""Base class for creating subsequent command classes and utility functions.

Classes
-------
BaseCommand

Functions
---------
get_command_class
    Return the class corresponding to the requested command.
"""


class BaseCommand(object):
    """Abstract command object that others should inherit from."""

    def run():
        return None
