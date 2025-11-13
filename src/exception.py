import sys
import traceback

class CustomException(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors
        _, _, tb = sys.exc_info()
        self.traceback = ''.join(traceback.format_stack()[:-1]) if tb is None else ''.join(traceback.format_exception(*sys.exc_info()))

    def __str__(self):
        return f"{self.args[0]}\nDetails: {self.errors}\nTraceback:\n{self.traceback}"
