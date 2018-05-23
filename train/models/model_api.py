import abc


class ModelAPI(object):
    @abc.abstractmethod
    def build_model(self, args=None):
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
