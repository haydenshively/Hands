from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    @abstractmethod
    def build(self):
        print("UNet constructed")
        pass

    @abstractmethod
    def compile(self):
        print("UNet compiled")
        pass
