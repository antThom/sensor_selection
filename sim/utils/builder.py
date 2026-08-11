"""Builder template for builder classes"""
from abc import ABC, abstractmethod
import functools


class BuilderTemplate(ABC):
    """Template bulider class for creating builders."""
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def build(self):
        pass
    
    def chainable(self, method):
        """
        Decorator to enable a function to be chained on others when calling the builder.
        """
        @functools.wraps(method)
        def wrapper(self):
            method(self)
            return self
        return wrapper
    

    