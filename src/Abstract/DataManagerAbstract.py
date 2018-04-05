import abc

class DataManagerAbstract(abc.ABC):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        print("DataReaderAbstract instance created.\n")
        super().__init__()

    @classmethod
    @abc.abstractmethod
    def loadPath(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def loadData(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def trainTestSplit(cls):
        raise NotImplementedError


    @classmethod
    @abc.abstractmethod
    def getData(cls):
        raise NotImplementedError
