from .create import CreateCommand
from .classifier import PickleCommand
from .evaluate import EvaluateCommand
from .upload import UploadCommand


REGISTERED_COMMANDS = {
    'create': CreateCommand,
    'upload': UploadCommand,
    'pickle': PickleCommand,
    'evaluate': EvaluateCommand
}


def get_command_class(**kws):
    for command in REGISTERED_COMMANDS:
        if kws[command]:
            return REGISTERED_COMMANDS[command](**kws)
    return None
