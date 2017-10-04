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
