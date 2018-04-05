import abc

class RecommenderModelAbstract(abc.ABC):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        print("RecommenderModelAbstract instance created.\n")
        super().__init__()

    @classmethod
    @abc.abstractmethod
    def fetchData(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def featureEngineering(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def dataSampling(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def dataFormatConversion(cls):
        raise NotImplementedError


    @classmethod
    @abc.abstractmethod
    def modelFitting(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def doPredictions(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def getPredictions(cls):
        raise NotImplementedError
