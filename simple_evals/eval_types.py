from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypedDict, Union
import typing

Message = TypedDict('Message', {'role': str, 'content': str, 'variant': Union[str, None]})
MessageList = list[Message]


class SamplerBase(ABC):
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """

    model: str
    _pack_message: Union[typing.Callable[[str, str], Message], None] = None

    @abstractmethod
    def __call__(self, messages: list[Message]) -> str:
        raise NotImplementedError


@dataclass
class EvalResult:
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: Union[float, None]  # top-line metric
    metrics: typing.Union[typing.Dict[str, float], None]  # other metrics
    htmls: list[str]  # strings of valid HTML
    convos: list[MessageList]  # sampled conversations


@dataclass
class SingleEvalResult:
    """
    Result of evaluating a single sample
    """

    score: Union[float, None]
    metrics: dict[str, Union[float, int, str]] = field(default_factory=dict)
    html: Union[str, None] = None
    convo: Union[MessageList, None] = None  # sampled conversation


class Eval(ABC):
    """
    Base class for defining an evaluation.
    """

    @abstractmethod
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError
