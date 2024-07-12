class ProcessError(Exception):
    def __init__(self, message):
        super(ProcessError, self).__init__(message)
        self.message = message
    def __str__(self):
        return self.message

class KeyError(Exception):
    def __init__(self, message):
        super(KeyError, self).__init__(message)
        self.message = message

    def __str__(self):
        return self.message