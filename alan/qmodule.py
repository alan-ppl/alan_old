import torch as t
import torch.nn as nn
from .utils import named2dim_tensor, extend_plates_with_named_tensor

class QModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.platedims = {}
        self._names = {}

    def get_unnamed_tensor(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        return None

    def get_named_tensor(self, name):
        tensor = self.get_unnamed_tensor(name)
        if tensor is not None:
            tensor = tensor.rename(*self._names[name])
        return tensor

    def __getattr__(self, name):
        tensor = self.get_named_tensor(name)
        if tensor is not None:
            return named2dim_tensor(self.platedims, tensor)
        else:
            return super().__getattr__(name)

    def register_buffer(self, name, tensor):
        self._names[name] = tensor.names
        self.platedims = extend_plates_with_named_tensor(tensor)
        return super().register_buffer(name, tensor.rename(None))

    def register_parameter(self, name, tensor):
        self._names[name] = tensor.names
        self.platedims = extend_plates_with_named_tensor(self.platedims, tensor)
        assert isinstance(tensor, nn.Parameter)
        tensor = nn.Parameter(tensor.rename(None))
        return super().register_parameter(name, tensor)

    def add_module(self, name, child):
        platedims = {**self.platedims}
        #Add new platedims from child
        for gc in child.modules():
            if isinstance(gc, QModule):
                platedims = {**platedims, **gc.platedims}
            else:
                for x in list(gc.parameters(recurse=False)) + list(gc.buffers(recurse=False)):
                    if any(name is not None in x.names):
                        raise Exception("Named parameter on an nn.Module.  To specify plates in approximate posteriors correctly, we need to use QModule in place of nn.Module")

        #Put the same platedims everywhere
        for mod in self.modules():
            if isinstance(mod, QModule):
                mod.platedims = platedims

        return super().add_module(name, mod)

    def __setattr__(self, name, tensor_module):
        if isinstance(tensor_module, t.Tensor):
            return self.register_parameter(name, tensor_module)
        elif isinstance(tensor_module, nn.Module):
            return self.add_module(name, tensor_module)
        else:
            return super().__setattr__(name, tensor_module)
