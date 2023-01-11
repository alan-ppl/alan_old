import torch.nn as nn
from .utils import named2dim_tensor, extend_plates_with_named_tensors
from .tensors import ValuedTraceTensor


def reconcile_platedims(self):
    if not isinstance(self, nn.Module):
        return {}

    all_platedims = {}
    for mod in self.modules():
        tensors = [*mod.parameters(recurse=False), *mod.buffers(recurse=False)]
        if isinstance(mod, QModule):
            all_platedims = extend_plates_with_named_tensors(all_platedims, tensors)
        else:
            for x in tensors:
                if any(name is not None in x.names):
                    raise Exception("Named parameter on an nn.Module.  To specify plates in approximate posteriors correctly, you should use alan.QModule in place of nn.Module.")

    for mod in self.modules():
        if isinstance(mod, QModule):
            mod._platedims = all_platedims

    return all_platedims

class QModule(nn.Module):
    """
    We try really hard to disconnect plate names from plate sizes.
    However, plate names and sizes are inextricably linked in
    typical approximate posteriors.  For instance, if we learn a
    mean for each datapoint, then we need one parameter for each
    datapoint.

    QModule mimics standard nn.Module, in the sense that all the 
    usual ways of defining parameters / submodules in __init__ work.
     
    The key difference is that QModule allows for named parameters.
    You provide named parameters using PyTorch named tensors. Then
    the QModule needs to make a torchdim for each named dimension.
    It does that, and ensures that the torchdims are consistent
    across nested QModules.

    Doesn't work if nn.Module has two QModule children.
    """
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
                raise Exception("Cannot return parameter or buffer, as self._platedims is not set.  To set self._platedims, you need to pass this module to a Model, or to sample_P.")
            return ValuedTraceTensor(named2dim_tensor(self._platedims, tensor))
        else:
            return super().__getattr__(name)

        for mod in self.modules():
            if isinstance(mod, QModule):
                mod._platedims = all_platedims

#    def register_buffer(self, name, tensor):
#        assert hasattr(mod, "platedims")
#        self.platedims = extend_plates_with_named_tensors(self.platedims, tensor)
#        return super().register_buffer(name, tensor)
#
#    def register_parameter(self, name, tensor):
#        assert hasattr(mod, "platedims")
#        self.platedims = extend_plates_with_named_tensors(self.platedims, tensor)
#        return super().register_parameter(name, tensor)
#
#    def __setattr__(self, name, tensor_module):
#        if isinstance(tensor_module, QModule):
#            for k in set(self.platedims.keys()).intersection(tensor_module.platedims.keys()):
#                assert self.platedims[k].size == tensor_module.platedims[k].size
#            all_platedims = {**self.platedims, **tensor_module.platedims}
#            self.platedims = all_platedims
#            tensor_module.platedims = all_platedims
#        elif isinstance(tensor_module, nn.Module):
#            #None of the submodules of an nn.Module can be a Qmodule
#            for mod in self.modules():
#                assert not isinstance(mod, QModule)
#            #None of the parameters of an nn.Module can be named
#            for x in [*mod.parameters(), *mod.buffers(recurse=False)]:
#                if any(name is not None in x.names):
#                    raise Exception("Named parameter on an nn.Module.  To specify plates in approximate posteriors correctly, we need to use QModule in place of nn.Module")
#        else:
#            self.register_parameter(self, name, tensor)
