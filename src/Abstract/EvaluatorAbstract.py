import abc

class EvaluatorAbstract(abc.ABC):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        print("EvaluatorAbstract instance created.\n")
        super().__init__()

    @classmethod
    @abc.abstractmethod
    def fetchData(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def prepareData(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def runScoring(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def showResults(cls):
        raise NotImplementedError
