from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    @abstractmethod
    def build(self):
        print("Constructed")
        pass

    @abstractmethod
    def compile(self):
        print("Compiled")
        pass
