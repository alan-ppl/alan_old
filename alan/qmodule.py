import torch.nn as nn
from .utils import named2dim_tensor

class QModule(nn.Module):
    def get_named_tensor(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        return None

    def __getattr__(self, name):
        tensor = self.get_named_tensor(name)
        if tensor is not None:
            if not hasattr(self, "_platedims"):
                raise Exception("Cannot return parameter or buffer, as self._platedims is not set.  To set self._platedims, you need to pass Q to a Model.")
            return named2dim_tensor(self._platedims, tensor)
        else:
            return super().__getattr__(name)

