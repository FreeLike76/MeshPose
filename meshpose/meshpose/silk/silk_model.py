import torch
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode, resize

import inspect
import itertools
from copy import deepcopy
from functools import partial
from heapq import heapify, heappop, heappush
from typing import Iterable, List, Set, Tuple, Union, Any, Dict, Callable, Optional
from collections import OrderedDict

import torch.nn.functional as F

class _Transition:
    def __init__(self, dependencies) -> None:
        # TODO check should be tuple
        self._dependencies = dependencies

    @property
    def dependencies(self):
        return self._dependencies

    def get_dependencies_from_session(self, session):
        return tuple(session[idx] for idx in self._dependencies)

    def __call__(self, session, inputs):
        raise NotImplementedError

class _InputExtraction(_Transition):
    def __init__(self, name) -> None:
        super().__init__(())
        self._name = name

    def __call__(self, _, inputs):
        return inputs[self._name]

class _ConstantExtraction(_Transition):
    def __init__(self, value) -> None:
        super().__init__(())
        self._value = value

    def __call__(self, _s, _i):
        return self._value

class _FunctionCall(_Transition):
    def __init__(
        self,
        function,
        *args,
        **kwargs,
    ) -> None:
        # TODO check arguments
        self._function = function
        ordered_keys = tuple(kwargs.keys())
        dependencies = tuple(args) + tuple(kwargs[key] for key in ordered_keys)

        self._n_args = len(args)
        self._key_to_index = {
            name: self._n_args + i for i, name in enumerate(ordered_keys)
        }
        self._signature = inspect.signature(function)

        # test bind
        self._signature.bind(*args, **kwargs)

        super().__init__(dependencies)

    def args(self, dependencies):
        return dependencies[: self._n_args]

    def kwargs(self, dependencies):
        return {name: dependencies[idx] for name, idx in self._key_to_index.items()}

    def __call__(self, session, _):
        dependency_values = self.get_dependencies_from_session(session)

        args = self.args(dependency_values)
        kwargs = self.kwargs(dependency_values)

        arguments = self._signature.bind(*args, **kwargs)
        arguments.apply_defaults()

        return self._function(*arguments.args, **arguments.kwargs)

class _TupleOutputExtraction(_Transition):
    def __init__(self, output_index, tuple_index) -> None:
        super().__init__((output_index,))
        self._tuple_index = tuple_index

    def __call__(self, session, _):
        # TODO check index bounds
        return self.get_dependencies_from_session(session)[0][self._tuple_index]

class FixedOutputFlow:
    def __init__(self, flow, outputs: Union[str, Tuple[str]]) -> None:
        self._flow = flow
        self._outputs = outputs

        self._tape = self._flow.get_tape(outputs)
        self._output_indexes = self._flow.names_to_indexes(self._outputs)

    @property
    def outputs(self):
        return self._outputs

    @property
    def tape(self):
        return self._tape

    @property
    def flow(self):
        return self._flow

    def __call__(self, *args, **kwargs):
        inputs = self._flow.inputs_as_dict(*args, **kwargs)
        return self._flow.flow_from_tape(self._tape, self._output_indexes, inputs)

    def with_outputs(self, outputs):
        return self.flow.with_outputs(outputs)

class Flow:
    class Constant:  # noqa: B903
        def __init__(self, value) -> None:
            self.value = value

    def __init__(self, *inputs: Tuple[str]) -> None:
        # TODO check redundant names
        # TODO check no "outputs" names
        # TODO should not be empty
        self._inputs = inputs
        self._name_to_index = {}
        self._index_to_name = {}
        self._transitions = []
        self._flow_signature = inspect.Signature(
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
            )
            for name in self._inputs
        )

        for name in self._inputs:
            self._add_transition(_InputExtraction(name), name)

    @property
    def inputs(self):
        return self._inputs

    @property
    def names(self):
        return tuple(self._index_to_name.values())

    def index_of(self, name):
        # TODO check if name exist
        if isinstance(name, str):
            return self._name_to_index[name]
        elif isinstance(name, Flow.Constant):
            return self._add_transition(_ConstantExtraction(name.value))
        raise RuntimeError(f"cannot handle name of type {type(name)}")

    def _add_transition(self, transition, name=None):
        # TODO check if name doesn't already exist
        index = len(self._transitions)
        self._transitions.append(transition)
        if name:
            self._name_to_index[name] = index
            self._index_to_name[index] = name
        return index

    def define_transition(
        self,
        names: Union[str, Tuple[str]],
        function,
        *args,
        **kwargs,
    ):
        # TODO check names
        args = tuple(self.index_of(name) for name in args)
        kwargs = {param: self.index_of(name) for param, name in kwargs.items()}

        transition = _FunctionCall(function, *args, **kwargs)

        if isinstance(names, str):
            index = self._add_transition(transition, name=names)
        else:
            index = self._add_transition(transition, name=None)
            for i, name in enumerate(names):
                self._add_transition(_TupleOutputExtraction(index, i), name=name)

    def get_tape(self, outputs):
        if isinstance(outputs, str):
            outputs = (outputs,)

        tape = []
        max_dependants = {}
        output_indexes = set(  # noqa: C401
            self._name_to_index[name] for name in outputs
        )

        head_indexes = [
            (-self._name_to_index[name], -self._name_to_index[name]) for name in outputs
        ]
        heapify(head_indexes)

        last_index = None
        while len(head_indexes) > 0:
            index, max_dependant = heappop(head_indexes)
            if index == last_index:
                continue
            last_index = index

            index = -index
            if max_dependant is not None:
                max_dependant = -max_dependant
                if index not in output_indexes:
                    max_dependants.setdefault(max_dependant, []).append(index)

            transition = self._transitions[index]
            for idx in transition.dependencies:
                heappush(head_indexes, (-idx, -index))

            tape.append(index)

        for i, index in enumerate(tape):
            tape[i] = (index, max_dependants.get(index, ()))

        return tuple(tape[::-1])

    def flow_from_tape(self, tape, output_indexes, inputs):
        session = [None] * len(self._transitions)
        for index, to_clean in tape:
            session[index] = self._transitions[index](session, inputs)
            for i in to_clean:
                session[i] = None

        if isinstance(output_indexes, int):
            return session[output_indexes]
        return tuple(session[index] for index in output_indexes)

    def names_to_indexes(self, names):
        if isinstance(names, str):
            return self._name_to_index[names]
        return tuple(self._name_to_index[name] for name in names)

    def inputs_as_dict(self, *args, **kwargs):
        return self._flow_signature.bind(*args, **kwargs).arguments

    def flow(self, outputs, *inputs_args, **inputs_kwargs):
        inputs = self.inputs_as_dict(*inputs_args, **inputs_kwargs)
        tape = self.get_tape(outputs)
        output_indexes = self.names_to_indexes(outputs)
        return self.flow_from_tape(tape, output_indexes, inputs)

    __call__ = flow

    def with_outputs(self, outputs):
        return FixedOutputFlow(self, outputs)

    def tape_as_pseudocode(self, tape):
        instructions = []
        for index, to_clean in tape:
            name = self._index_to_name.get(index, "@")
            transition = self._transitions[index]
            if isinstance(transition, _FunctionCall):
                dep = tuple(
                    self._index_to_name.get(index, "@")
                    for index in transition.dependencies
                )
                args = transition.args(dep)
                kwargs = {k: v for k, v in transition.kwargs(dep).items()}
                all_args = itertools.chain(args, kwargs)
                all_args = ",".join(all_args)
                func_name = getattr(
                    transition._function, "__name__", repr(transition._function)
                )
                instructions.append(f"{name} = {func_name}({all_args})")
            elif isinstance(transition, _InputExtraction):
                instructions.append(f"${transition._name}")
            elif isinstance(transition, _TupleOutputExtraction):
                instructions.append(f"{name} = @[{transition._tuple_index}]")

            for i in to_clean:
                name = self._index_to_name.get(i, "@")
                instructions.append(f"delete {name}")

        return "\n".join(instructions)


class AutoForward:
    def __init__(self, flow: Flow, default_outputs: Union[str, Tuple[str]]) -> None:
        self._default_outputs = default_outputs
        self._flow = flow
        self._forward_flow = None

    @property
    def default_outputs(self):
        return self._default_outputs

    @property
    def flow(self):
        return self._flow

    def forward_flow(self, outputs: Union[str, Tuple[str]], *args, **kwargs):
        return self._flow(outputs, *args, **kwargs)

    def forward(self, *args, **kwargs):
        if self._forward_flow is None:
            self._forward_flow = self._flow.with_outputs(self._default_outputs)
        return self._forward_flow(*args, **kwargs)


class CoordinateMapping:
    def apply(self, positions):
        raise NotImplementedError

    def reverse(self, positions):
        raise NotImplementedError

    def __add__(self, other):
        return SequentialCoordinateMapping((self, other))

    def __neg__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

class SequentialCoordinateMapping(CoordinateMapping):
    def __init__(self, mappings: Iterable[CoordinateMapping]) -> None:
        super().__init__()
        self.mappings = tuple(mappings)

    def apply(self, positions):
        for mapping in self.mappings:
            positions = mapping.apply(positions)
        return positions

    def reverse(self, positions):
        for mapping in reversed(self.mappings):
            positions = mapping.reverse(positions)
        return positions

    def __radd__(self, other):
        if isinstance(other, SequentialCoordinateMapping):
            return SequentialCoordinateMapping(other.mappings + self.mappings)
        return SequentialCoordinateMapping((other,) + self.mappings)

    def __neg__(self):
        return SequentialCoordinateMapping(reversed(self.mappings))

    def __str__(self):
        return " <- ".join(f"({str(mapping)})" for mapping in reversed(self.mappings))

class CoordinateMappingComposer:
    def __init__(self) -> None:
        self._mappings = {}
        self._arrows = set()

    def _set(self, id_from, id_to, mapping):
        if (id_to, id_from) in self._arrows:
            raise RuntimeError(f"the mapping '{id_to}' <- '{id_from}' already exist")

        m = self._mappings.setdefault(id_to, {})
        m[id_from] = mapping

        m = self._mappings.setdefault(id_from, {})
        m[id_to] = -mapping

        self._arrows.add((id_to, id_from))
        self._arrows.add((id_from, id_to))

    def set(self, id_from, id_to, mapping: CoordinateMapping):
        if not isinstance(mapping, CoordinateMapping):
            raise RuntimeError(
                f"the provided mapping should subclass `CoordinateMapping` to provide coordinate mapping between {id_from} and {id_to}"
            )

        for node_id in self._mappings.get(id_from, {}):
            self._set(node_id, id_to, self._mappings[id_from][node_id] + mapping)

        self._set(id_from, id_to, mapping)

    def get(self, id_from, id_to):
        return self._mappings[id_to][id_from]


class CoordinateMappingProvider:
    def mappings(self) -> Tuple[CoordinateMapping]:
        raise NotImplementedError

# TODO(Pierre): Overload list specific methods as we need them.
class MixedModuleList(torch.nn.Module):
    """Works the same as `torch.nn.ModuleList`, but allows to have non-module items."""

    def __init__(self, items: Iterable[Any]) -> None:
        super().__init__()

        self._mods = torch.nn.ModuleList(
            [mod for mod in items if isinstance(mod, torch.nn.Module)]
        )
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx: int):
        return self._items[idx]

class MixedModuleDict(torch.nn.Module):
    """Works the same as `torch.nn.ModuleDict`, but allows to have non-module items."""

    def __init__(self, items: Dict[Any, Any] = None) -> None:
        super().__init__()

        items = OrderedDict() if items is None else items
        self._mods = torch.nn.ModuleDict(
            {key: mod for key, mod in items.items() if isinstance(mod, torch.nn.Module)}
        )
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, key: Any):
        return self._items[key]

    def __setitem__(self, key: str, item: Any) -> None:
        if key in self._mods:
            del self._mods[key]
        if isinstance(item, torch.nn.Module):
            self._mods[key] = item
        self._items[key] = item

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def items(self):
        return self._items.items()

    def keys(self):
        return self._items.keys()

class SharedBackboneMultipleHeads(
    AutoForward,
    torch.nn.Module,
):
    def __init__(
        self,
        backbone,
        input_name: str,
        backbone_output_name: Union[str, Tuple[str]],
    ) -> None:
        torch.nn.Module.__init__(self)
        AutoForward.__init__(
            self,
            Flow(input_name),
            default_outputs=backbone_output_name,
        )

        self._input_name = input_name
        self._backbone_output_name = backbone_output_name
        self._backbone = backbone

        # handle coordinate mappings
        self._coordinate_mappings_composer = CoordinateMappingComposer()
        assert isinstance(self._backbone, CoordinateMappingProvider)

        if self.is_multi_backbone_outputs:
            assert len(self._backbone_output_name) == len(self._backbone.mappings())

            for i, mapping in enumerate(self._backbone.mappings()):
                self._coordinate_mappings_composer.set(
                    self._input_name,
                    self._backbone_output_name[i],
                    mapping,
                )
        else:
            self._coordinate_mappings_composer.set(
                self._input_name,
                self._backbone_output_name,
                self._backbone.mappings(),
            )

        # self._coordinate_mappings: Dict[str, CoordinateMapping],
        self.flow.define_transition(backbone_output_name, self.backbone, input_name)

        self._heads = MixedModuleDict()

    @property
    def coordinate_mapping_composer(self):
        return self._coordinate_mappings_composer

    @property
    def backbone(self):
        return self._backbone

    @property
    def is_multi_backbone_outputs(self):
        return not isinstance(self.backbone_output_name, str)

    @property
    def heads(self):
        return self._heads

    @property
    def head_names(self):
        return tuple(self._heads.keys())

    @property
    def backbone_output_name(self):
        return self._backbone_output_name

    @property
    def input_name(self):
        return self._input_name

    def add_head_to_backbone_output(self, head_name, head, backbone_output_name=None):
        if backbone_output_name is None:
            if self.is_multi_backbone_outputs:
                raise RuntimeError(
                    f"the backbone has {len(self.backbone_output_name)} outputs {self.backbone_output_name} and one should be set using the `backbone_output_name` parameter"
                )
            backbone_output_name = self.backbone_output_name
        elif not isinstance(backbone_output_name, str):
            raise RuntimeError(
                "invalid type for `backbone_output_name` parameter, should be a string"
            )

        # check existing head
        if head_name in self.heads:
            raise RuntimeError(f"head '{head_name}' has already been added")

        if not isinstance(head, CoordinateMappingProvider):
            raise RuntimeError(
                f"head '{head_name}' should sub-class `CoordinateMappingProvider`"
            )

        self.heads[head_name] = head
        self.flow.define_transition(head_name, head, backbone_output_name)
        self._coordinate_mappings_composer.set(
            backbone_output_name,
            head_name,
            head.mappings(),
        )

    def add_head(self, head_name, head):
        self.add_head_to_backbone_output(head_name, head)

    def add_heads(self, **heads):
        # add heads and make them accessible to flow
        for name, head in heads.items():
            self.add_head(name, head)



class SiLKBase(AutoForward, torch.nn.Module):
    def __init__(
        self,
        backbone,
        input_name: str = "images",
        backbone_output_name: Union[str, Tuple[str]] = "features",
        default_outputs: Union[str, Iterable[str]] = ("descriptors", "score"),
    ):
        torch.nn.Module.__init__(self)

        self.backbone = SharedBackboneMultipleHeads(
            backbone=backbone,
            input_name=input_name,
            backbone_output_name=backbone_output_name,
        )

        self.detector_heads = set()
        self.descriptor_heads = set()

        AutoForward.__init__(self, self.backbone.flow, default_outputs=default_outputs)

    @property
    def coordinate_mapping_composer(self):
        return self.backbone.coordinate_mapping_composer

    def add_detector_head(self, head_name, head, backbone_output_name=None):
        self.backbone.add_head_to_backbone_output(head_name, head, backbone_output_name)
        self.detector_heads.add(head_name)

    def add_descriptor_head(self, head_name, head, backbone_output_name=None):
        self.backbone.add_head_to_backbone_output(head_name, head, backbone_output_name)
        self.descriptor_heads.add(head_name)



def vgg_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    use_batchnorm: bool = True,
    non_linearity: str = "relu",
    padding: int = 1,
) -> torch.nn.Module:
    """
    The VGG block for the model.
    This block contains a 2D convolution, a ReLU activation, and a
    2D batch normalization layer.
    Args:
        in_channels (int): the number of input channels to the Conv2d layer
        out_channels (int): the number of output channels
        kernel_size (int): the size of the kernel for the Conv2d layer
        use_batchnorm (bool): whether or not to include a batchnorm layer.
            Default is true (batchnorm will be used).
    Returns:
        vgg_blk (nn.Sequential): the vgg block layer of the model
    """

    if non_linearity == "relu":
        non_linearity = torch.nn.ReLU(inplace=True)
    else:
        raise NotImplementedError

    # the paper states that batchnorm is used after each convolution layer
    if use_batchnorm:
        vgg_blk = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            non_linearity,
            torch.nn.BatchNorm2d(out_channels),
        )
    # however, the official implementation does not include batchnorm
    else:
        vgg_blk = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            non_linearity,
        )

    return vgg_blk


class VGG(torch.nn.Module, CoordinateMappingProvider):
    """
    The VGG backbone.
    """

    def __init__(
        self,
        num_channels: int = 1,
        use_batchnorm: bool = False,
        use_max_pooling: bool = True,
        padding: int = 1,
    ):
        """
        Initialize the VGG backbone model.
        Can take an input image of any number of channels (e.g. grayscale, RGB).
        """
        torch.nn.Module.__init__(self)
        CoordinateMappingProvider.__init__(self)

        assert padding in {0, 1}

        self.padding = padding
        self.use_max_pooling = use_max_pooling

        if use_max_pooling:
            self.mp = torch.nn.MaxPool2d(2, stride=2)
        else:
            self.mp = torch.nn.Identity()

        # convolution layers (encoder)
        self.l1 = torch.nn.Sequential(
            vgg_block(
                num_channels,
                64,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
            vgg_block(
                64,
                64,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
        )

        self.l2 = torch.nn.Sequential(
            vgg_block(
                64,
                64,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
            vgg_block(
                64,
                64,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
        )

        self.l3 = torch.nn.Sequential(
            vgg_block(
                64,
                128,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
            vgg_block(
                128,
                128,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
        )

        self.l4 = torch.nn.Sequential(
            vgg_block(
                128,
                128,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
            vgg_block(
                128,
                128,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
        )

    def mappings(self):
        mapping = Identity()
        mapping = mapping + mapping_from_torch_module(self.l1)
        mapping = mapping + mapping_from_torch_module(self.mp)
        mapping = mapping + mapping_from_torch_module(self.l2)
        mapping = mapping + mapping_from_torch_module(self.mp)
        mapping = mapping + mapping_from_torch_module(self.l3)
        mapping = mapping + mapping_from_torch_module(self.mp)
        mapping = mapping + mapping_from_torch_module(self.l4)

        return mapping

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Goes through the layers of the VGG model as the forward pass.
        Computes the output.
        Args:
            images (tensor): image pytorch tensor with
                shape N x num_channels x H x W
        Returns:
            output (tensor): the output point pytorch tensor with
            shape N x cell_size^2+1 x H/8 x W/8.
        """
        o1 = self.l1(images)
        o1 = self.mp(o1)

        o2 = self.l2(o1)
        o2 = self.mp(o2)

        o3 = self.l3(o2)
        o3 = self.mp(o3)

        # features
        o4 = self.l4(o3)

        return o4


def parametric_vgg_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    normalization_fn,
    non_linearity: str = "relu",
    padding: int = 1,
) -> torch.nn.Module:
    if non_linearity == "relu":
        non_linearity = torch.nn.ReLU(inplace=True)
    else:
        raise NotImplementedError

    vgg_blk = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        non_linearity,
        normalization_fn,
    )

    return vgg_blk


class ParametricVGG(torch.nn.Module, CoordinateMappingProvider):
    DEFAULT_NORMALIZATION_FN = torch.nn.Identity()

    def __init__(
        self,
        input_num_channels: int = 1,
        normalization_fn: Union[Callable, List[Callable]] = DEFAULT_NORMALIZATION_FN,
        use_max_pooling: bool = True,
        padding: int = 1,
        channels: List[int] = (64, 64, 128, 128),
    ):
        CoordinateMappingProvider.__init__(self)
        torch.nn.Module.__init__(self)

        assert padding in {0, 1}
        assert len(channels) >= 1

        self.padding = padding
        self.use_max_pooling = use_max_pooling
        if isinstance(normalization_fn, Iterable):
            normalization_fn = tuple(normalization_fn)
            assert len(normalization_fn) == len(channels)
        else:
            normalization_fn = tuple([normalization_fn] * len(channels))

        if use_max_pooling:
            self.mp = torch.nn.MaxPool2d(2, stride=2)
        else:
            self.mp = torch.nn.Identity()

        self.layers = []
        self.channels = (input_num_channels,) + channels
        for i in range(1, len(self.channels)):
            layer = torch.nn.Sequential(
                parametric_vgg_block(
                    self.channels[i - 1],
                    self.channels[i],
                    3,
                    deepcopy(normalization_fn[i - 1]),
                    "relu",
                    padding,
                ),
                parametric_vgg_block(
                    self.channels[i],
                    self.channels[i],
                    3,
                    deepcopy(normalization_fn[i - 1]),
                    "relu",
                    padding,
                ),
            )
            self.layers.append(layer)
        self.layers = torch.nn.ModuleList(self.layers)

    def mappings(self):
        mapping = Identity()
        for layer in self.layers[:-1]:
            mapping = mapping + mapping_from_torch_module(layer)
            mapping = mapping + mapping_from_torch_module(self.mp)
        mapping = mapping + mapping_from_torch_module(self.layers[-1])

        return mapping

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.mp(x)
        x = self.layers[-1](x)
        return x


Backbone = partial(
    VGG,
    num_channels=1,
    use_batchnorm=True,
    use_max_pooling=True,
)

class Identity(CoordinateMapping):
    def apply(self, positions):
        return positions

    def reverse(self, positions):
        return positions

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __neg__(self):
        return self

    def __str__(self):
        return "x <- x"

class LinearCoordinateMapping(CoordinateMapping):
    def __init__(self, scale=1.0, bias=0.0) -> None:
        super().__init__()
        self.scale = scale
        self.bias = bias

    def apply(self, positions):
        device = (
            positions.device if isinstance(positions, torch.torch.Tensor) else "cpu"
        )
        return positions * self.scale.to(device) + self.bias.to(device)

    def reverse(self, positions):
        device = (
            positions.device if isinstance(positions, torch.torch.Tensor) else "cpu"
        )
        return (positions - self.bias.to(device)) / self.scale.to(device)

    def __add__(self, other):
        if isinstance(other, LinearCoordinateMapping):
            return LinearCoordinateMapping(
                self.scale * other.scale,
                self.bias * other.scale + other.bias,
            )
        elif isinstance(other, Identity):
            return self
        return CoordinateMapping.__add__(self, other)

    def __neg__(self):
        return LinearCoordinateMapping(
            scale=1.0 / self.scale,
            bias=-self.bias / self.scale,
        )

    def __str__(self):
        return f"x <- {self.scale} x + {self.bias}"

class Conv2dCoordinateMapping(LinearCoordinateMapping):
    @staticmethod
    def from_conv_module(module):
        assert (
            isinstance(module, torch.nn.Conv2d)
            or isinstance(module, torch.nn.MaxPool2d)
            or isinstance(module, torch.nn.ConvTranspose2d)
        )
        if isinstance(module, torch.nn.ConvTranspose2d):
            return -Conv2dCoordinateMapping(
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
            )
        return Conv2dCoordinateMapping(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
        )

    def __init__(self, kernel_size, stride=1, padding=0, dilation=1) -> None:
        # TODO(Pierre) : Generalize later if necessary
        assert dilation == 1 or dilation == (1, 1)

        kernel_size = torch.tensor(kernel_size)
        stride = torch.tensor(stride)
        padding = torch.tensor(padding)

        output_coord_to_input_coord = LinearCoordinateMapping(
            scale=stride,
            bias=-0.5 * stride - padding + kernel_size / 2,
        )
        input_coord_to_output_coord = -output_coord_to_input_coord

        LinearCoordinateMapping.__init__(
            self,
            input_coord_to_output_coord.scale,
            input_coord_to_output_coord.bias,
        )

def mapping_from_torch_module(module) -> CoordinateMapping:
    if isinstance(module, CoordinateMappingProvider):
        return module.mappings()
    elif isinstance(module, torch.nn.Conv2d):
        return Conv2dCoordinateMapping.from_conv_module(module)
    elif isinstance(module, torch.nn.ConvTranspose2d):
        return Conv2dCoordinateMapping.from_conv_module(module)
    elif isinstance(module, torch.nn.modules.pooling.MaxPool2d):
        return Conv2dCoordinateMapping.from_conv_module(module)
    elif isinstance(module, torch.nn.Sequential):
        return sum((mapping_from_torch_module(mod) for mod in module), Identity())
    elif (
        isinstance(module, torch.nn.modules.activation.ReLU)
        or isinstance(module, torch.nn.modules.activation.LeakyReLU)
        or isinstance(module, torch.nn.Identity)
        or isinstance(module, torch.nn.BatchNorm2d)
        or isinstance(module, torch.nn.InstanceNorm2d)
        or isinstance(module, torch.nn.GroupNorm)
    ):
        return Identity()
    else:
        raise RuntimeError(
            f"cannot get the coordinate mappings of module of type {type(module)}"
        )

class DetectorHead(torch.nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        in_channels: int = 128,
        lat_channels: int = 256,
        out_channels: int = 1,
        use_batchnorm: bool = True,
        padding: int = 1,
        detach: bool = False,
    ) -> None:
        torch.nn.Module.__init__(self)
        CoordinateMappingProvider.__init__(self)

        assert padding in {0, 1}

        self._detach = detach

        self._detH1 = vgg_block(
            in_channels,
            lat_channels,
            3,
            use_batchnorm=use_batchnorm,
            padding=padding,
        )

        if use_batchnorm:
            # no relu (bc last layer) - option to have batchnorm or not
            self._detH2 = nn.Sequential(
                nn.Conv2d(lat_channels, out_channels, 1, padding=0),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # if no batch norm
            self._detH2 = nn.Sequential(
                nn.Conv2d(lat_channels, out_channels, 1, padding=0),
            )

    def mappings(self):
        mapping = mapping_from_torch_module(self._detH1)
        mapping = mapping + mapping_from_torch_module(self._detH2)
        return mapping

    def forward(self, x: torch.Tensor):
        if self._detach:
            x = x.detach()

        x = self._detH1(x)
        x = self._detH2(x)
        return x

class DescriptorHead(torch.nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 256,
        use_batchnorm: bool = True,
        padding: int = 1,
    ) -> None:
        torch.nn.Module.__init__(self)
        CoordinateMappingProvider.__init__(self)

        assert padding in {0, 1}

        # descriptor head (decoder)
        self._desH1 = vgg_block(
            in_channels,
            out_channels,
            3,
            use_batchnorm=use_batchnorm,
            padding=padding,
        )

        if use_batchnorm:
            # no relu (bc last layer) - option to have batchnorm or not
            self._desH2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, padding=0),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # if no batch norm - note that normailzation is calculated later
            self._desH2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, padding=0),
            )

    def mappings(self):
        mapping = mapping_from_torch_module(self._desH1)
        mapping = mapping + mapping_from_torch_module(self._desH2)
        return mapping

    def forward(self, x: torch.Tensor):
        x = self._desH1(x)
        x = self._desH2(x)
        return x


def logits_to_prob(logits: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
    """
    Get the probabilities given the logits output.

    The probabilities are the chances that each of the points
    in the image is a corner.

    Args:
        logits (tensor): the logits output from the model

    Returns:
        prob (tensor): the probabilities tensor (not reshaped) with shape
            (batch_size, 65, H/8, W/8) for cell_size = 8
    """

    # the logits tensor size is batch_size, 65, img_height, img_width
    # 65 is 8 * 8 + 1, where cell_size = 8
    channels = logits.shape[channel_dim]
    if channels == 1:
        prob = torch.sigmoid(logits)
    else:
        prob = torch.softmax(logits, dim=channel_dim)
    return prob


def depth_to_space(
    prob: torch.Tensor,
    cell_size: int = 8,
    channel_dim: int = 1,
) -> torch.Tensor:
    """
    Reorganizes output shape to be of size batch_size x img_height x img_width.

    Converts the structure of the outputs from a tensor consisting
    of a series of cells of size cell_size x cell_size, where
    cells correspond to groups of pixels in the image, to a tensor
    where the shape corresponds exactly to the shape of the image.

    Args:
        prob (tensor): the tensor comprising the corner probabilities for each
            pixel in a "depth" format as a tensor of cell_size x cell_size cells
        cell_size (int): the size of each cell (default is always 8)

    Returns:
        image_probs (tensor): the reshaped tensor where each image in the batch
            is shaped in a tensor of size 1 x H x W
    """
    if cell_size > 1:
        assert prob.shape[channel_dim] == cell_size * cell_size + 1

        # remove the last (dustbin) cell from the list
        prob, _ = prob.split(cell_size * cell_size, dim=channel_dim)

        # change the dimensions to get an output shape of (batch_size, H, W)
        image_probs = F.pixel_shuffle(prob, cell_size)
    else:
        assert prob.shape[channel_dim] == 1
        image_probs = prob

    return image_probs

def remove_border_points(image_nms: torch.Tensor, border_dist: int = 4) -> torch.Tensor:
    """
    Remove predicted points within border_dist pixels of the image border.

    Args:
        image_nms (tensor): the output of the nms function, a tensor of shape
            (img_height, img_width) with corner probability values at each pixel location
        border_dist (int): the distance from the border to remove points

    Returns:
        image_nms (tensor): the image with all probability values equal to 0.0
            for pixel locations within border_dist of the image border
    """
    if border_dist > 0:
        # left columns
        image_nms[..., :, :border_dist] = 0.0

        # right columns
        image_nms[..., :, -border_dist:] = 0.0

        # top rows
        image_nms[..., :border_dist, :] = 0.0

        # bottom rows
        image_nms[..., -border_dist:, :] = 0.0

    return image_nms

def fast_nms(
    image_probs: torch.Tensor,
    nms_dist: int = 4,
    max_iter: int = -1,
    min_value: float = 0.0,
) -> torch.Tensor:
    """Produce same result as `original_nms` (see documentation).
    The process is slightly different :
      1. Find any local maximum (and count them).
      2. Suppress their neighbors (by setting them to 0).
      3. Repeat 1. and 2. until the number of local maximum stays the same.

    Performance
    -----------
    The original implementation takes about 2-4 seconds on a batch of 32 images of resolution 240x320.
    This fast implementation takes about ~90ms on the same input.

    Parameters
    ----------
    image_probs : torch.Tensor
        Tensor of shape BxCxHxW.
    nms_dist : int, optional
        The minimum distance between two predicted corners after NMS, by default 4
    max_iter : int, optional
        Maximum number of iteration, by default -1.
        Setting this number to a positive integer guarantees execution speed, but not correctness (i.e. good approximation).
    min_value : float
        Minimum value used for suppression.

    Returns
    -------
    torch.Tensor
        Tensor of shape BxCxHxW containing NMS suppressed input.
    """
    if nms_dist == 0:
        return image_probs

    ks = 2 * nms_dist + 1
    midpoint = (ks * ks) // 2
    count = None
    batch_size = image_probs.shape[0]

    i = 0
    while True:
        if i == max_iter:
            break

        # get neighbor probs in last dimension
        unfold_image_probs = F.unfold(
            image_probs,
            kernel_size=(ks, ks),
            dilation=1,
            padding=nms_dist,
            stride=1,
        )
        unfold_image_probs = unfold_image_probs.reshape(
            batch_size,
            ks * ks,
            image_probs.shape[-2],
            image_probs.shape[-1],
        )

        # check if middle point is local maximum
        max_idx = unfold_image_probs.argmax(dim=1, keepdim=True)
        mask = max_idx == midpoint

        # count all local maximum that are found
        new_count = mask.sum()

        # we stop if we din't not find any additional local maximum
        if new_count == count:
            break
        count = new_count

        # propagate local-maximum information to local neighbors (to suppress them)
        mask = mask.float()
        mask = mask.expand(-1, ks * ks, -1, -1)
        mask = mask.view(batch_size, ks * ks, -1)
        mask = mask.contiguous()
        mask[:, midpoint] = 0.0  # make sure we don't suppress the local maximum itself
        fold_ = F.fold(
            mask,
            output_size=image_probs.shape[-2:],
            kernel_size=(ks, ks),
            dilation=1,
            padding=nms_dist,
            stride=1,
        )

        # suppress all points who have a local maximum in their neighboorhood
        image_probs = image_probs.masked_fill(fold_ > 0.0, min_value)

        i += 1

    return image_probs


def original_nms(image_probs: torch.Tensor, nms_dist: int = 4) -> torch.Tensor:
    """
    Run non-maximum suppression on the predicted corner points.

    NMS removes nearby points within a distance of nms_dist from the point
    with the highest probability value in that region. The algorithm is as
    follows:
        1. Order the predicted corner points from highest to lowest probability.
        2. Set up the input tensor with probability values for each pixel location
        to have padding of size nms_dist so points near the border can be suppressed.
        3. Go through each point in the list from step 1. If the point has not already
        been suppressed in the probability value tensor with padding from step 2
        (meaning the probability value has been changed to 0.0), suppress all
        points within nms_dist from that point by changing their probability values
        to 0.0. Keep the probability value of the current point as is.
        3. At the end, remove the padding from the tensor. Thus, the output is
        a tensor of size (img_height, img_width) with probability values for the remaining
        predicted corner pixels (those not suppressed) and 0.0 for non-corner pixels.

    Args:
        image_probs (tensor): a tensor of size (img_height, img_width) where each
            pixel location has value equal to the probability value of it being a corner,
            as predicted by the model.
        nms_dist (int): the minimum distance between two predicted corners after NMS

    Returns:
        image_probs_nms (tensor): a tensor of size (img_height, img_width) where each
            pixel location has value equal to the probability value of it being a corner,
            after running the non-maximum suppression algorithm. Thus, no two predicted
            corners will be within nms_dist pixels of each other.
    """
    # each elem in corners_list is (row, col) for predicted corners in image_probs
    corners_list = torch.nonzero(image_probs)

    # list of the probability values in the same order as the list of their locations
    list_of_prob = image_probs[torch.nonzero(image_probs, as_tuple=True)]

    # concatenate the probability values with their locations (prob, row, col) for each
    corners_list_with_prob = torch.cat(
        (list_of_prob.unsqueeze(dim=1), corners_list), dim=1
    )

    # sort the list of probability values with most confident corners first
    prob_indices = torch.argsort(list_of_prob, dim=0, descending=True)

    # sort the list of corner locations according to the order of the indices
    sorted_corners_list = corners_list_with_prob[prob_indices]

    # pad the border of the grid with zeros, so that we can NMS points near the border
    padding = (nms_dist, nms_dist, nms_dist, nms_dist)
    padded_image_probs = F.pad(image_probs, padding, "constant", 0)

    # go through each element in the sorted list of corners
    # suppress surrounding points by converting their probabilities to 0.0
    # TODO: Benchmark this to see if this loop is a bottleneck
    for prob, row, col in sorted_corners_list:
        row = int(row)
        col = int(col)

        # if the point hasn't already been suppressed
        if padded_image_probs[row + nms_dist][col + nms_dist] != 0.0:
            # suppress all points in the (2*nms_dist, 2*nms_dist) square around the point
            padded_image_probs[
                row : row + 2 * nms_dist + 1,  # noqa: E203
                col : col + 2 * nms_dist + 1,  # noqa: E203
            ] = 0.0

            # then add back in the one point not suppressed
            padded_image_probs[row + nms_dist][col + nms_dist] = prob

    # remove the image padding to get the actual image size
    image_probs_nms = padded_image_probs[nms_dist:-nms_dist, nms_dist:-nms_dist]

    return image_probs_nms

def prob_map_to_points_map(
    prob_map: torch.Tensor,
    prob_thresh: float = 0.015,
    nms_dist: int = 4,
    border_dist: int = 4,
    use_fast_nms: bool = True,
    top_k: int = None,
):
    prob_map = remove_border_points(prob_map, border_dist=border_dist)

    prob_map = prob_map.squeeze(dim=1)

    prob_thresh = torch.tensor(prob_thresh, device=prob_map.device)
    prob_thresh = prob_thresh.unsqueeze(0)

    if use_fast_nms:
        # add missing channel
        prob_map = prob_map.unsqueeze(1)
        nms = fast_nms(prob_map, nms_dist=nms_dist)
        # remove added channel
        prob_map = nms.squeeze(1)
    else:
        # Original Implementation
        # NMS only runs one image at a time, so go through each elem in the batch
        prob_map = torch.stack(
            [original_nms(image, nms_dist=nms_dist) for image in prob_map]
        )

    if top_k:
        if top_k >= prob_map.shape[-1] * prob_map.shape[-2]:
            top_k_threshold = torch.zeros_like(prob_thresh)
        else:
            # infer top k threshold
            top_k = torch.tensor(top_k, device=prob_map.device)
            reshaped_prob_map = prob_map.reshape(prob_map.shape[0], -1)

            top_k_percentile = (
                reshaped_prob_map[0].size()[0] - top_k - 1
            ) / reshaped_prob_map[0].size()[0]

            top_k_threshold = reshaped_prob_map.quantile(
                top_k_percentile,
                dim=1,
                interpolation="midpoint",
            )
        prob_thresh = torch.minimum(top_k_threshold, prob_thresh)
        prob_thresh = prob_thresh.unsqueeze(-1).unsqueeze(-1)

    # only take points with probability above the probability threshold
    prob_map = torch.where(
        prob_map > prob_thresh,
        prob_map,
        torch.tensor(0.0, device=prob_map.device),
    )

    return prob_map  # batch_output


def prob_map_to_positions_with_prob(
    prob_map: torch.Tensor,
    threshold: float = 0.0,
) -> Tuple[torch.Tensor]:
    """Convert probability map to positions with probability associated with each position.

    Parameters
    ----------
    prob_map : torch.Tensor
        Probability map. Tensor of size N x 1 x H x W.
    threshold : float, optional
        Threshold used to discard positions with low probability, by default 0.0

    Returns
    -------
    Tuple[Tensor]
        Tuple of positions (with probability) tensors of size P x 3 (x, y and prob).
    """
    prob_map = prob_map.squeeze(dim=1)
    positions = tuple(
        torch.nonzero(prob_map[i] > threshold).float() + 0.5
        for i in range(prob_map.shape[0])
    )
    prob = tuple(
        prob_map[i][torch.nonzero(prob_map[i] > threshold, as_tuple=True)][:, None]
        for i in range(prob_map.shape[0])
    )
    positions_with_prob = tuple(
        torch.cat((pos, prob), dim=1) for pos, prob in zip(positions, prob)
    )
    return positions_with_prob


class MagicPoint(AutoForward, torch.nn.Module):
    def __init__(
        self,
        *,
        use_batchnorm: bool = True,
        num_channels: int = 1,
        cell_size: int = 8,
        detection_threshold=0.015,
        detection_top_k=None,
        nms_dist=4,
        border_dist=4,
        use_max_pooling: bool = True,
        input_name="images",
        backbone=None,
        backbone_output_name: str = "features",
        detector_head=None,
        detector_head_output_name: str = "logits",
        default_outputs=None,
    ):
        torch.nn.Module.__init__(self)

        # architecture parameters
        self._num_channels = num_channels
        self._cell_size = cell_size  # depends on VGG's downsampling

        # detection parameters
        self._detection_threshold = detection_threshold
        self._detection_top_k = detection_top_k
        self._nms_dist = nms_dist
        self._border_dist = border_dist

        # add backbone
        self.backbone = SharedBackboneMultipleHeads(
            backbone=Backbone(
                num_channels=num_channels,
                use_batchnorm=use_batchnorm,
                use_max_pooling=use_max_pooling,
            )
            if backbone is None
            else backbone,
            input_name=input_name,
            backbone_output_name=backbone_output_name,
        )

        if use_max_pooling:
            out_channels = cell_size * cell_size + 1
        else:
            out_channels = 1

        # add detector head
        self.backbone.add_head(
            detector_head_output_name,
            DetectorHead(
                in_channels=128,
                lat_channels=256,
                out_channels=out_channels,
                use_batchnorm=use_batchnorm,
            )
            if detector_head is None
            else detector_head,
        )

        # add the forward function
        default_outputs = (
            (backbone_output_name, detector_head_output_name)
            if default_outputs is None
            else default_outputs
        )
        AutoForward.__init__(self, self.backbone.flow, default_outputs)

        # add detector head post-processing
        MagicPoint.add_detector_head_post_processing(
            self.flow,
            detector_head_output_name=detector_head_output_name,
            prefix="",
            cell_size=self._cell_size,
            detection_threshold=self._detection_threshold,
            detection_top_k=self._detection_top_k,
            nms_dist=self._nms_dist,
            border_dist=self._border_dist,
        )

    @staticmethod
    def add_detector_head_post_processing(
        flow: Flow,
        detector_head_output_name: str = "logits",
        prefix: str = "magicpoint.",
        cell_size: int = 8,
        detection_threshold=0.015,
        detection_top_k=None,
        nms_dist=4,
        border_dist=4,
    ):
        flow.define_transition(
            f"{prefix}probability",
            logits_to_prob,
            detector_head_output_name,
        )
        flow.define_transition(
            f"{prefix}score",
            partial(depth_to_space, cell_size=cell_size),
            f"{prefix}probability",
        )
        flow.define_transition(
            f"{prefix}nms",
            partial(
                prob_map_to_points_map,
                prob_thresh=detection_threshold,
                nms_dist=nms_dist,
                border_dist=border_dist,
                top_k=detection_top_k,
            ),
            f"{prefix}score",
        )
        flow.define_transition(
            f"{prefix}positions",
            prob_map_to_positions_with_prob,
            f"{prefix}nms",
        )


class SuperPoint(AutoForward, torch.nn.Module):
    """
    The SuperPoint model, as a subclass of the MagicPoint model.
    """

    def __init__(
        self,
        *,
        use_batchnorm: bool = True,
        descriptor_scale_factor: float = 1.0,
        input_name: str = "images",
        descriptor_head=None,
        descriptor_head_output_name="raw_descriptors",
        default_outputs=("coarse_descriptors", "logits"),
        **magicpoint_kwargs,
    ):
        """Initialize the SuperPoint model.

        Assumes an RGB image with 1 color channel (grayscale image).

        Parameters
        ----------
        use_batchnorm : bool, optional
            Specify if the model uses batch normalization, by default True
        """
        torch.nn.Module.__init__(self)

        self._descriptor_scale_factor = descriptor_scale_factor
        self.magicpoint = MagicPoint(
            use_batchnorm=use_batchnorm,
            input_name=input_name,
            **magicpoint_kwargs,
        )

        AutoForward.__init__(self, self.magicpoint.flow, default_outputs)

        self.magicpoint.backbone.add_head(
            descriptor_head_output_name,
            DescriptorHead(
                in_channels=128, out_channels=256, use_batchnorm=use_batchnorm
            )
            if descriptor_head is None
            else descriptor_head,
        )

        SuperPoint.add_descriptor_head_post_processing(
            self.flow,
            input_name=input_name,
            descriptor_head_output_name=descriptor_head_output_name,
            prefix="",
            scale_factor=self._descriptor_scale_factor,
        )

    @staticmethod
    def add_descriptor_head_post_processing(
        flow: Flow,
        input_name: str = "images",
        descriptor_head_output_name: str = "raw_descriptors",
        prefix: str = "superpoint.",
        scale_factor: float = 1.0,
    ):
        flow.define_transition(
            f"{prefix}coarse_descriptors",
            partial(SuperPoint.normalize_descriptors, scale_factor=scale_factor),
            descriptor_head_output_name,
        )
        flow.define_transition(f"{prefix}image_size", SuperPoint.image_size, input_name)
        flow.define_transition(
            f"{prefix}sparse_descriptors",
            partial(SuperPoint.sparsify_descriptors, scale_factor=scale_factor),
            descriptor_head_output_name,
            f"{prefix}positions",
            f"{prefix}image_size",
        )
        flow.define_transition(
            f"{prefix}upsampled_descriptors",
            partial(SuperPoint.upsample_descriptors, scale_factor=scale_factor),
            descriptor_head_output_name,
            f"{prefix}image_size",
        )

    @staticmethod
    def image_size(images):
        return images.shape[-2:]

    @staticmethod
    def normalize_descriptors(raw_descriptors, scale_factor=1.0, normalize=True):
        if normalize:
            return scale_factor * F.normalize(
                raw_descriptors,
                p=2,
                dim=1,
            )  # L2 normalization
        return scale_factor * raw_descriptors

    @staticmethod
    def sparsify_descriptors(
        raw_descriptors,
        positions,
        image_size,
        scale_factor: float = 1.0,
    ):
        image_size = torch.tensor(
            image_size,
            dtype=torch.float,
            device=raw_descriptors.device,
        )
        sparse_descriptors = []

        for i, pos in enumerate(positions):
            pos = pos[:, :2]
            n = pos.shape[0]

            # handle edge case when no points has been detected
            if n == 0:
                desc = raw_descriptors[i]
                fdim = desc.shape[0]
                sparse_descriptors.append(
                    torch.zeros(
                        (n, fdim),
                        dtype=desc.dtype,
                        device=desc.device,
                    )
                )
                continue

            # revert pixel centering for grad sample
            pos = pos - 0.5

            # normalize to [-1. +1] & prepare for grid sample
            pos = 2.0 * (pos / (image_size - 1)) - 1.0
            pos = pos[:, [1, 0]]
            pos = pos[None, None, ...]

            # process descriptor output by interpolating into descriptor map using 2D point locations\
            # note that grid_sample takes coordinates in x, y order (col, then row)
            descriptors = raw_descriptors[i][None, ...]
            descriptors = F.grid_sample(
                descriptors,
                pos,
                mode="bilinear",
                align_corners=False,
            )
            descriptors = descriptors.view(-1, n).T

            # L2 normalize the descriptors
            descriptors = SuperPoint.normalize_descriptors(descriptors, scale_factor)

            sparse_descriptors.append(descriptors)
        return sparse_descriptors

    @staticmethod
    def upsample_descriptors(raw_descriptors, image_size, scale_factor: float = 1.0):
        upsampled_descriptors = resize(
            raw_descriptors,
            image_size,
            interpolation=InterpolationMode.BILINEAR,
        )
        return SuperPoint.normalize_descriptors(upsampled_descriptors, scale_factor)


class HomographicSampler:
    """Samples multiple homographic crops from multiples batched images.

    This sampler makes it very easy to sample homographic crops from multiple images by manipulating a virtual crop initially centered on the entire image.
    Applying successive simple transformations (xyz-rotation, shift, scale) will modify the position and shape of that virtual crop.
    Transformations operates on normalized coordinates independent of an image shape.
    The initial virtual crop has its top-left position at (-1, -1), and bottom-right position at (+1, +1).
    Thus the center being at position (0, 0).

    Examples
    --------

    ```python
    hc = HomographicSampler(2, "cpu") # homographic sampler with 2 virtual crops

    hc.scale(0.5) # reduce all virtual crops size by half
    hc.shift(((-0.25, -0.25), (+0.25, +0.25))) # shift first virtual crop to top-left part, second virtual crop to bottom-right part
    hc.rotate(3.14/4., axis="x", clockwise=True, local_center=True) # rotate both virtual crops locally by 45 degrees clockwise (around x-axis)

    crops = hc.extract_crop(image, (100, 100)) # extract two homographic crops defined earlier as (100, 100) images
    ```

    """

    _DEST_COORD = torch.tensor(
        [
            [-1.0, -1.0],  # top-left
            [+1.0, -1.0],  # top-right
            [-1.0, +1.0],  # bottom-left
            [+1.0, +1.0],  # bottom-right
        ],
        dtype=torch.double,
    )

    _VALID_AXIS = {"x", "y", "z"}
    _VALID_DIRECTIONS = {"forward", "backward"}
    _VALID_ORDERING = {"xy", "yx"}

    def __init__(self, batch_size: int, device: str):
        """

        Parameters
        ----------
        batch_size : int
            Number of virtual crops to handle.
        device : str
            Device on which operations will be done.
        """
        self.reset(batch_size, device)

    @staticmethod
    def _convert_points_from_homogeneous(
        points: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Function that converts points from homogeneous to Euclidean space."""

        # we check for points at max_val
        z_vec: torch.Tensor = points[..., -1:]

        # set the results of division by zeror/near-zero to 1.0
        # follow the convention of opencv:
        # https://github.com/opencv/opencv/pull/14411/files
        mask: torch.Tensor = torch.abs(z_vec) > eps
        scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

        return scale * points[..., :-1]

    @staticmethod
    def _convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
        """Function that converts points from Euclidean to homogeneous space."""

        return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)

    @staticmethod
    def _transform_points(
        trans_01: torch.Tensor, points_1: torch.Tensor
    ) -> torch.Tensor:
        """Function that applies a linear transformations to a set of points."""

        points_1 = points_1.to(trans_01.device)
        points_1 = points_1.to(trans_01.dtype)

        # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
        shape_inp = points_1.shape
        points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
        trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])
        # We expand trans_01 to match the dimensions needed for bmm
        trans_01 = torch.repeat_interleave(
            trans_01, repeats=points_1.shape[0] // trans_01.shape[0], dim=0
        )
        # to homogeneous
        points_1_h = HomographicSampler._convert_points_to_homogeneous(
            points_1
        )  # BxNxD+1
        # transform coordinates
        points_0_h = torch.bmm(points_1_h, trans_01.permute(0, 2, 1))
        points_0_h = torch.squeeze(points_0_h, dim=-1)
        # to euclidean
        points_0 = HomographicSampler._convert_points_from_homogeneous(
            points_0_h
        )  # BxNxD
        # reshape to the input shape
        points_0 = points_0.reshape(shape_inp)
        return points_0

    @staticmethod
    def _create_meshgrid(
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        normalized: bool = True,
    ) -> torch.Tensor:
        """Generate a coordinate grid for an image."""
        if normalized:
            min_x = -1.0
            max_x = +1.0
            min_y = -1.0
            max_y = +1.0
        else:
            min_x = 0.5
            max_x = width - 0.5
            min_y = 0.5
            max_y = height - 0.5

        xs: torch.Tensor = torch.linspace(
            min_x,
            max_x,
            width,
            device=device,
            dtype=dtype,
        )
        ys: torch.Tensor = torch.linspace(
            min_y,
            max_y,
            height,
            device=device,
            dtype=dtype,
        )

        # generate grid by stacking coordinates
        base_grid: torch.Tensor = torch.stack(
            torch.meshgrid([xs, ys], indexing="ij"), dim=-1
        )  # WxHx2
        return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2

    @staticmethod
    def _build_perspective_param(
        p: torch.Tensor, q: torch.Tensor, axis: str
    ) -> torch.Tensor:
        ones = torch.ones_like(p)[..., 0:1]
        zeros = torch.zeros_like(p)[..., 0:1]
        if axis == "x":
            return torch.cat(
                [
                    p[:, 0:1],
                    p[:, 1:2],
                    ones,
                    zeros,
                    zeros,
                    zeros,
                    -p[:, 0:1] * q[:, 0:1],
                    -p[:, 1:2] * q[:, 0:1],
                ],
                dim=1,
            )

        if axis == "y":
            return torch.cat(
                [
                    zeros,
                    zeros,
                    zeros,
                    p[:, 0:1],
                    p[:, 1:2],
                    ones,
                    -p[:, 0:1] * q[:, 1:2],
                    -p[:, 1:2] * q[:, 1:2],
                ],
                dim=1,
            )

        raise NotImplementedError(
            f"perspective params for axis `{axis}` is not implemented."
        )

    @staticmethod
    def _get_perspective_transform(src, dst):
        r"""Calculate a perspective transform from four pairs of the corresponding
        points.

        The function calculates the matrix of a perspective transform so that:

        .. math ::

            \begin{bmatrix}
            t_{i}x_{i}^{'} \\
            t_{i}y_{i}^{'} \\
            t_{i} \\
            \end{bmatrix}
            =
            \textbf{map_matrix} \cdot
            \begin{bmatrix}
            x_{i} \\
            y_{i} \\
            1 \\
            \end{bmatrix}

        where

        .. math ::
            dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3

        Args:
            src: coordinates of quadrangle vertices in the source image with shape :math:`(B, 4, 2)`.
            dst: coordinates of the corresponding quadrangle vertices in
                the destination image with shape :math:`(B, 4, 2)`.

        Returns:
            the perspective transformation with shape :math:`(B, 3, 3)`.
        """

        # we build matrix A by using only 4 point correspondence. The linear
        # system is solved with the least square method, so here
        # we could even pass more correspondence
        p = []
        for i in [0, 1, 2, 3]:
            p.append(
                HomographicSampler._build_perspective_param(src[:, i], dst[:, i], "x")
            )
            p.append(
                HomographicSampler._build_perspective_param(src[:, i], dst[:, i], "y")
            )

        # A is Bx8x8
        A = torch.stack(p, dim=1)

        # b is a Bx8x1
        b = torch.stack(
            [
                dst[:, 0:1, 0],
                dst[:, 0:1, 1],
                dst[:, 1:2, 0],
                dst[:, 1:2, 1],
                dst[:, 2:3, 0],
                dst[:, 2:3, 1],
                dst[:, 3:4, 0],
                dst[:, 3:4, 1],
            ],
            dim=1,
        )

        # solve the system Ax = b
        X = torch.linalg.solve(A, b)

        # create variable to return
        batch_size = src.shape[0]
        M = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
        M[..., :8] = torch.squeeze(X, dim=-1)

        return M.view(-1, 3, 3)  # Bx3x3

    def reset(self, batch_size: Optional[int] = None, device: Optional[str] = None):
        """Resets all the crops to their initial position and sizes.

        Parameters
        ----------
        batch_size : int, optional
            Number of virtual crops to handle, by default None.
        device : str, optional
            Device on which operations will be done, by default None.
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        device = self.device if device is None else device

        self._dest_coords = HomographicSampler._DEST_COORD.to(device)
        self._dest_coords = self._dest_coords.unsqueeze(0)
        self._dest_coords = self._dest_coords.expand(batch_size, -1, -1)

        self._homog_src_coords = HomographicSampler._convert_points_to_homogeneous(
            self._dest_coords
        )

        self._clear_cache()

    def _clear_cache(self):
        """Intermediate data are cached such that the same homographic sampler can efficiently be called several times using the same homographic transforms."""
        self._src_coords = None
        self._forward_matrices = None
        self._backward_matrices = None

    def _to(self, device, name):
        attr = getattr(self, name)
        if attr is not None:
            setattr(self, name, attr.to(device))

    def to(self, device: str):
        """Moves all operations to new device.

        Parameters
        ----------
        device : str
            Pytorch device.
        """
        if device != self.device:
            self._to(device, "_dest_coords")
            self._to(device, "_src_coords")
            self._to(device, "_homog_src_coords")
            self._to(device, "_forward_matrices")
            self._to(device, "_backward_matrices")
        return self

    @property
    def batch_size(self):
        return self._homog_src_coords.shape[0]

    @property
    def device(self):
        return self._homog_src_coords.device

    @property
    def dtype(self):
        return self._homog_src_coords.dtype

    @property
    def src_coords(self) -> torch.Tensor:
        """Coordinates of the homographic crop corners in the virtual image coordinate reference system.
        Those four points are ordered as : (top-left, top-right, bottom-left, bottom-right)

        Returns
        -------
        torch.Tensor
            :math:`(B, 4, 2)` tensor containing the homographic crop foud corners coordinates.
        """
        if self._src_coords is None:
            self._src_coords = HomographicSampler._convert_points_from_homogeneous(
                self._homog_src_coords
            )
        return self._src_coords

    @property
    def dest_coords(self) -> torch.Tensor:
        return self._dest_coords

    def _auto_expand(self, input, outer_dim_size=None, **kwargs):
        """Auto-expand scalar or iterables to be batched."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, **kwargs)

        # scalar
        if len(input.shape) == 0:
            input = input.unsqueeze(0)
            if outer_dim_size is None:
                outer_dim_size = 1
            else:
                input = input.expand(outer_dim_size)

        # vector
        if len(input.shape) == 1:
            if outer_dim_size is None:
                outer_dim_size = input.shape[0]
            elif outer_dim_size != input.shape[0]:
                raise RuntimeError(
                    f"provided outer dim size {outer_dim_size} doesn't match input shape {input.shape}"
                )

            input = input.unsqueeze(0)
            input = input.expand(self.batch_size, -1)

        if len(input.shape) != 2:
            raise RuntimeError(f"input should have size BxD (shape is {input.shape}")

        input = input.type(self.dtype)
        input = input.to(self.device)

        return input

    def rotate(
        self,
        angles: Union[float, torch.Tensor],
        clockwise: bool = False,
        axis: str = "z",
        local_center: bool = False,
    ):
        """Rotate virtual crops.

        Parameters
        ----------
        angles : Union[float, torch.Tensor]
            Angles of rotation. If scalar, applied to all crops. If :math:`(B, 1)` tensor, applied to each crop independently.
        clockwise : bool, optional
            Rotational direction, by default False
        axis : str, optional
            Axis of rotation, by default "z". Valid values are "x", "y" and "z". "z" is in-plane rotation. "x" and "y" are out-of-plane rotations.
        local_center : bool, optional
            Rotate on the center of the crop, by default False. If False, use global center of rotation (i.e. initial crop center). This option is only relevant after a shift has been used.

        Raises
        ------
        RuntimeError
            Raised if provided axis is invalid.
        """
        if axis not in HomographicSampler._VALID_AXIS:
            raise RuntimeError(
                f'provided axis "{axis}" isn\'t valid, should be one of {HomographicSampler._VALID_AXIS}'
            )

        angles = self._auto_expand(angles, outer_dim_size=1)

        if clockwise:
            angles = -angles

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        _1 = torch.ones_like(cos_a)
        _0 = torch.zeros_like(cos_a)

        if axis == "z":
            flatmat = [+cos_a, -sin_a, _0, +sin_a, +cos_a, _0, _0, _0, _1]
        elif axis == "y":
            flatmat = [+cos_a, _0, -sin_a, _0, _1, _0, +sin_a, _0, +cos_a]
        elif axis == "x":
            flatmat = [_1, _0, _0, _0, +cos_a, +sin_a, _0, -sin_a, +cos_a]

        rot_matrix = torch.cat(flatmat, dim=-1)
        rot_matrix = rot_matrix.view(self.batch_size, 3, 3)

        self._clear_cache()

        if local_center:
            center = torch.mean(self._homog_src_coords, dim=1, keepdim=True)

            self._homog_src_coords -= center
            self._homog_src_coords = self._homog_src_coords @ rot_matrix
            self._homog_src_coords += center
        else:
            if axis != "z":
                self._homog_src_coords[..., -1] -= 1.0
            self._homog_src_coords = self._homog_src_coords @ rot_matrix
            if axis != "z":
                self._homog_src_coords[..., -1] += 1.0

    def shift(self, delta: Union[float, Tuple[float, float], torch.Tensor]):
        """Shift virtual crops.

        Parameters
        ----------
        delta : Union[float, Tuple[float, float], torch.Tensor]
            Shift values. Scalar or Tuple will be applied to all crops. :math:`(B, 2)` tensors will be applied to each crop independently.
        """

        delta = self._auto_expand(delta, outer_dim_size=2)
        delta = delta.unsqueeze(1)
        delta = delta * self._homog_src_coords[..., -1].unsqueeze(-1)

        self._clear_cache()
        self._homog_src_coords[..., :2] += delta

    def scale(
        self,
        factors: Union[float, Tuple[float, float], torch.Tensor],
        local_center: bool = False,
    ):
        """Scale the virtual crops.

        Parameters
        ----------
        factors : Union[float, Tuple[float, float], torch.Tensor]
            Scaling factors. Scalar or Tuple will be applied to all crops. :math:`(B, 2)` tensors will be applied to each crop independently.
        local_center : bool, optional
            Scale on the center of the crop, by default False. If False, use global center of rotation (i.e. initial crop center). This option is only relevant after a shift has been used.
        """
        factors = self._auto_expand(factors, outer_dim_size=2)
        factors = factors.unsqueeze(1)

        self._clear_cache()

        if local_center:
            center = torch.mean(self._homog_src_coords, dim=1, keepdim=True)

            self._homog_src_coords -= center
            self._homog_src_coords[..., :2] *= factors
            self._homog_src_coords += center
        else:
            self._homog_src_coords[..., :2] *= factors

    @property
    def forward_matrices(self):
        if self._forward_matrices is None:
            self._forward_matrices = HomographicSampler._get_perspective_transform(
                self.dest_coords,
                self.src_coords,
            )
        return self._forward_matrices

    @property
    def backward_matrices(self):
        if self._backward_matrices is None:
            self._backward_matrices = HomographicSampler._get_perspective_transform(
                self.src_coords,
                self.dest_coords,
            )
        return self._backward_matrices

    def extract_crop(
        self,
        images: torch.Tensor,
        sampling_size: Tuple[int, int],
        mode="bilinear",
        padding_mode="zeros",
        direction="forward",
    ) -> torch.Tensor:
        """Extract all crops from a set of images.

        It can handle one-image-to-many-crops and many-images-to-many-crops.
        If the number of images is smaller than the number of crops, a number of n crops will be asssigned to each image such that :math:`n_crops = n * n_images`.

        Parameters
        ----------
        images : torch.Tensor
            Tensor containing all images (valid shapes are :math:`(B,C,H,W)` and :math:`(C,H,W)`).
        sampling_size : Tuple[int, int]
            Spatial shape of the output crops.
        mode : str, optional
            Sampling mode passed to `grid_sample`, by default "bilinear".
        padding_mode : str, optional
            Padding mode passed to `grid_sample`, by default "zeros".
        direction : str, optional
            Direction of the crop sampling (`src -> dest` or `dest -> src`), by default "forward". Valid are "forward" and "backward".

        Returns
        -------
        torch.Tensor
            Sampled crops using transformed virtual crops.

        Raises
        ------
        RuntimeError
            Raised is `images` shape is invalid.
        RuntimeError
            Raised is `images` batch size isn't a multiple of the number of virtual crops.
        RuntimeError
            Raised is `direction` is invalid.
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)
        elif images.dim() != 4:
            raise RuntimeError("provided image(s) should be of shape BxCxHxW or CxHxW")

        if self.batch_size % images.shape[0] != 0:
            raise RuntimeError(
                f"the sampler batch size ({self.batch_size}) should be a multiple of the image batch size (found {images.shape[0]})"
            )

        if direction not in HomographicSampler._VALID_DIRECTIONS:
            raise RuntimeError(
                f'invalid direction "{direction}" found, should be one of {self._VALID_DIRECTIONS}'
            )

        # reshape images to handle multiple crops
        crop_per_image = self.batch_size // images.shape[0]
        images = images.unsqueeze(1)
        images = images.expand(-1, crop_per_image, -1, -1, -1)
        images = images.reshape(self.batch_size, *images.shape[2:])

        # select homography matrix
        if direction == "forward":
            matrix = self.forward_matrices
        else:
            matrix = self.backward_matrices

        # create grid of coordinates used for image sampling
        grid = HomographicSampler._create_meshgrid(
            sampling_size[0],
            sampling_size[1],
            device=matrix.device,
            dtype=matrix.dtype,
        )
        grid = grid.expand(self.batch_size, -1, -1, -1)
        grid = HomographicSampler._transform_points(matrix[:, None, None], grid)
        grid = grid.type_as(images)

        # sample pixels using transformed grid coordinates
        return nn.functional.grid_sample(
            images,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        )

    def transform_points(
        self,
        points: Union[torch.Tensor, List[torch.Tensor]],
        image_shape: Optional[Tuple[int, int]] = None,
        direction: str = "forward",
        ordering: str = "xy",
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Apply homography to a set of points.

        Parameters
        ----------
        points : Union[torch.Tensor, List[torch.Tensor]]
            BxNx2 tensor or list of Nx2 tensors containing the coordinates to transform.
        image_shape : Optional[Tuple[int, int]], optional
            Shape of the tensor the coordinates references, as in (height, width), by default None.
            If not provided, the coordinates are assumed to be already normalized between [-1, +1].
        direction : str, optional
            Direction of the homography, by default "forward".
        ordering : str, optional
            Specify the order in which the x,y coordinates are stored in "points", by default "xy".

        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            Transformed coordinates.

        Raises
        ------
        RuntimeError
            If the provided direction is invalid.
        RuntimeError
            If the provided ordering is invalid.
        """
        # check arguments
        if direction not in HomographicSampler._VALID_DIRECTIONS:
            raise RuntimeError(
                f'invalid direction "{direction}" found, should be one of {self._VALID_DIRECTIONS}'
            )
        if ordering not in HomographicSampler._VALID_ORDERING:
            raise RuntimeError(
                f'invalid ordering "{ordering}" found, should be one of {self._VALID_ORDERING}'
            )

        # select homography matrices
        if direction == "forward":
            matrix = self.backward_matrices
        else:
            matrix = self.forward_matrices

        # pad input if using variable length
        lengths = None
        if not isinstance(points, torch.Tensor):
            lengths = [p.shape[0] for p in points]
            points = nn.utils.rnn.pad_sequence(points, batch_first=True)

        # convert to "xy" ordering
        if ordering == "yx":
            points = points[..., [1, 0]]

        # bring coordinates to [-1, +1] range
        if image_shape is not None:
            image_shape = torch.tensor(
                [image_shape[1], image_shape[0]],
                dtype=points.dtype,
                device=points.device,
            )
            image_shape = image_shape[None, None, ...]
            image_shape_half = image_shape / 2.0
            pixel_shift = 0.5 / image_shape
            points = (points - image_shape_half) / image_shape_half + pixel_shift

        # reshape points to handle multiple transforms
        transform_per_points = self.batch_size // points.shape[0]
        points = points.unsqueeze(1)
        points = points.expand(-1, transform_per_points, -1, -1)
        points = points.reshape(self.batch_size, *points.shape[2:])

        # change lengths size accordingly
        if transform_per_points != 1:
            lengths = list(
                itertools.chain.from_iterable(
                    itertools.repeat(s, transform_per_points) for s in lengths
                )
            )

        # apply homography to point coordinates
        transformed_points = HomographicSampler._transform_points(
            matrix[:, None, None], points
        )

        # bring coordinates to original range
        if image_shape is not None:
            transformed_points = (
                (transformed_points - pixel_shift) * image_shape_half
            ) + image_shape_half

        # convert back to initial ordering
        if ordering == "yx":
            transformed_points = transformed_points[..., [1, 0]]

        # remove padded results if input was variable length
        if lengths is not None:
            transformed_points = [
                transformed_points[i, :s] for i, s in enumerate(lengths)
            ]

        return transformed_points

def get_dense_positions(h, w, device, batch_size=None):
    dense_positions = HomographicSampler._create_meshgrid(
        w,
        h,
        device=device,
        normalized=False,
    )
    dense_positions = dense_positions.permute(0, 2, 1, 3)
    dense_positions = dense_positions.reshape(-1, 2)
    dense_positions = dense_positions.unsqueeze(0)

    if batch_size is not None:
        dense_positions = dense_positions.expand(batch_size, -1, -1)

    return dense_positions

class SiLKVGG(SiLKBase):
    def __init__(
        self,
        in_channels,
        *,
        feat_channels: int = 128,
        lat_channels: int = 128,
        desc_channels: int = 128,
        use_batchnorm: bool = True,
        backbone=None,
        detector_head=None,
        descriptor_head=None,
        detection_threshold: float = 0.8,
        detection_top_k: int = 100,
        nms_dist=4,
        border_dist=4,
        descriptor_scale_factor: float = 1.0,
        learnable_descriptor_scale_factor: bool = False,
        normalize_descriptors: bool = True,
        padding: int = 1,
        **base_kwargs,
    ) -> None:
        backbone = (
            Backbone(
                num_channels=in_channels,
                use_batchnorm=use_batchnorm,
                use_max_pooling=False,
                padding=padding,
            )
            if backbone is None
            else backbone
        )

        detector_head = (
            DetectorHead(
                in_channels=feat_channels,
                lat_channels=lat_channels,
                out_channels=1,
                use_batchnorm=use_batchnorm,
                padding=padding,
            )
            if detector_head is None
            else detector_head
        )

        descriptor_head = (
            DescriptorHead(
                in_channels=feat_channels,
                out_channels=desc_channels,
                use_batchnorm=use_batchnorm,
                padding=padding,
            )
            if descriptor_head is None
            else descriptor_head
        )

        SiLKBase.__init__(
            self,
            backbone=backbone,
            **base_kwargs,
        )

        self.add_detector_head("logits", detector_head)
        self.add_descriptor_head("raw_descriptors", descriptor_head)

        self.descriptor_scale_factor = nn.parameter.Parameter(
            torch.tensor(descriptor_scale_factor),
            requires_grad=learnable_descriptor_scale_factor,
        )
        self.normalize_descriptors = normalize_descriptors

        MagicPoint.add_detector_head_post_processing(
            self.flow,
            "logits",
            prefix="",
            cell_size=1,
            detection_threshold=detection_threshold,
            detection_top_k=detection_top_k,
            nms_dist=nms_dist,
            border_dist=border_dist,
        )

        SiLKVGG.add_descriptor_head_post_processing(
            self.flow,
            input_name=self.backbone.input_name,
            descriptor_head_output_name="raw_descriptors",
            prefix="",
            scale_factor=self.descriptor_scale_factor,
            normalize_descriptors=normalize_descriptors,
        )

    @staticmethod
    def add_descriptor_head_post_processing(
        flow: Flow,
        input_name: str = "images",
        descriptor_head_output_name: str = "raw_descriptors",
        positions_name: str = "positions",
        prefix: str = "superpoint.",
        scale_factor: float = 1.0,
        normalize_descriptors: bool = True,
    ):
        flow.define_transition(
            f"{prefix}normalized_descriptors",
            partial(
                SuperPoint.normalize_descriptors,
                scale_factor=scale_factor,
                normalize=normalize_descriptors,
            ),
            descriptor_head_output_name,
        )
        flow.define_transition(
            f"{prefix}dense_descriptors",
            SiLKVGG.get_dense_descriptors,
            f"{prefix}normalized_descriptors",
        )
        flow.define_transition(f"{prefix}image_size", SuperPoint.image_size, input_name)
        flow.define_transition(
            f"{prefix}sparse_descriptors",
            partial(
                SiLKVGG.sparsify_descriptors,
                scale_factor=scale_factor,
                normalize_descriptors=normalize_descriptors,
            ),
            descriptor_head_output_name,
            positions_name,
        )
        flow.define_transition(
            f"{prefix}sparse_positions",
            lambda x: x,
            positions_name,
        )
        flow.define_transition(
            f"{prefix}dense_positions",
            SiLKVGG.get_dense_positions,
            "probability",
        )

    @staticmethod
    def get_dense_positions(probability):
        batch_size = probability.shape[0]
        device = probability.device
        dense_positions = get_dense_positions(
            probability.shape[2],
            probability.shape[3],
            device,
            batch_size=batch_size,
        )

        dense_probability = probability.reshape(probability.shape[0], -1, 1)
        dense_positions = torch.cat((dense_positions, dense_probability), dim=2)

        return dense_positions

    @staticmethod
    def get_dense_descriptors(normalized_descriptors):
        dense_descriptors = normalized_descriptors.reshape(
            normalized_descriptors.shape[0],
            normalized_descriptors.shape[1],
            -1,
        )
        dense_descriptors = dense_descriptors.permute(0, 2, 1)
        return dense_descriptors

    @staticmethod
    def sparsify_descriptors(
        raw_descriptors,
        positions,
        scale_factor: float = 1.0,
        normalize_descriptors: bool = True,
    ):
        sparse_descriptors = []
        for i, pos in enumerate(positions):
            pos = pos[:, :2]
            pos = pos.floor().long()

            descriptors = raw_descriptors[i, :, pos[:, 0], pos[:, 1]].T

            # L2 normalize the descriptors
            descriptors = SuperPoint.normalize_descriptors(
                descriptors,
                scale_factor,
                normalize_descriptors,
            )

            sparse_descriptors.append(descriptors)
        return tuple(sparse_descriptors)


def from_feature_coords_to_image_coords(model, desc_positions):
    if isinstance(desc_positions, tuple):
        return tuple(
            from_feature_coords_to_image_coords(
                model,
                dp,
            )
            for dp in desc_positions
        )
    coord_mapping = model.coordinate_mapping_composer.get("images", "raw_descriptors")
    desc_positions = torch.cat(
        [
            coord_mapping.reverse(desc_positions[..., :2]),
            desc_positions[..., 2:],
        ],
        dim=-1,
    )

    return desc_positions