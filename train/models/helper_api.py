import abc


class HelperAPI(object):
    @abc.abstractmethod
    def build_model(self, args=None, for_training=True):
        pass

    @abc.abstractmethod
    def preprocess_input(self, input):
        pass

    @abc.abstractmethod
    def preprocess_target(self, target):
        pass

    @abc.abstractmethod
    def postprocess_output(self, output):
        pass

    @abc.abstractmethod
    def monitor_val(self):
        return "val_loss"

    @abc.abstractmethod
    def monitor_mode(self):
        return "auto"
