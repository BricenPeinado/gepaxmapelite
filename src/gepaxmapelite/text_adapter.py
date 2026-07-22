"""A small, provider-neutral GEPA adapter for text cleanup experiments.

The adapter deliberately owns no model clients.  Callers inject a task executor and
an output judge, which makes every GEPA inner run easy to isolate and test.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol


@dataclass(frozen=True, slots=True)
class TextExample:
    """One task-model input used by GEPA's training or selection validation set."""

    text: str
    example_id: str = ""
    reference: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TextJudgment:
    """A task output score and the textual feedback GEPA will reflect on."""

    score: float
    feedback: str
    objective_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("TextJudgment.score must be finite and between 0 and 1")
        for name, value in self.objective_scores.items():
            if not name:
                raise ValueError("objective score names cannot be empty")
            if not math.isfinite(value):
                raise ValueError(f"objective score {name!r} must be finite")


@dataclass(frozen=True, slots=True)
class TextTrajectory:
    """Opaque execution evidence later converted into GEPA reflective examples."""

    example: TextExample
    output: str
    judgment: TextJudgment


class TaskExecutor(Protocol):
    """Run a downstream task prompt on one input string."""

    def __call__(self, task_prompt: str, text: str) -> str: ...


class OutputJudge(Protocol):
    """Score a task-model output and provide reflection feedback."""

    def __call__(self, example: TextExample, output: str) -> TextJudgment: ...


class TextCleanupAdapter:
    """GEPA adapter that evolves one named downstream text-cleanup prompt.

    ``propose_new_texts`` must remain ``None``.  That tells GEPA to use its own
    reflective proposer, where the Promptbreeder-evolved meta-prompt is injected as
    ``reflection_prompt_template`` by :class:`GEPAInnerRunner`.
    """

    propose_new_texts: ClassVar[None] = None

    def __init__(
        self,
        executor: TaskExecutor,
        judge: OutputJudge,
        *,
        prompt_component: str = "task_prompt",
        recoverable_exceptions: tuple[type[Exception], ...] = (),
    ) -> None:
        if not prompt_component:
            raise ValueError("prompt_component cannot be empty")
        self._executor = executor
        self._judge = judge
        self.prompt_component = prompt_component
        exceptions = tuple(recoverable_exceptions)
        if any(
            not isinstance(exc_type, type) or not issubclass(exc_type, Exception)
            for exc_type in exceptions
        ):
            raise TypeError("recoverable_exceptions must contain Exception subclasses")
        self._recoverable_exceptions = exceptions

    def evaluate(
        self,
        batch: list[TextExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        """Execute and judge a GEPA candidate on a batch.

        Exceptions propagate by default so authentication, provider, and client
        failures cannot masquerade as ordinary zero scores. Exception types listed
        in ``recoverable_exceptions`` become score-zero reflective trajectories.
        The GEPA import remains lazy, keeping archive-only use lightweight.
        """

        try:
            task_prompt = candidate[self.prompt_component]
        except KeyError as exc:
            raise ValueError(
                f"candidate is missing prompt component {self.prompt_component!r}"
            ) from exc

        outputs: list[str] = []
        scores: list[float] = []
        trajectories: list[TextTrajectory] = []
        objectives: list[dict[str, float]] = []

        for example in batch:
            output = ""
            try:
                output = self._executor(task_prompt, example.text)
                if not isinstance(output, str):
                    raise TypeError("task executor must return str")
                judgment = self._judge(example, output)
                if not isinstance(judgment, TextJudgment):
                    raise TypeError("output judge must return TextJudgment")
            except self._recoverable_exceptions as exc:
                judgment = TextJudgment(
                    score=0.0,
                    feedback=f"Execution or judging failed: {type(exc).__name__}: {exc}",
                )

            outputs.append(output)
            scores.append(judgment.score)
            objectives.append(dict(judgment.objective_scores))
            if capture_traces:
                trajectories.append(TextTrajectory(example, output, judgment))

        from gepa.core.adapter import EvaluationBatch

        objective_scores = objectives if any(objectives) else None
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            objective_scores=objective_scores,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Turn captured task traces into JSON-safe GEPA side information."""

        trajectories = eval_batch.trajectories
        if trajectories is None:
            raise ValueError("GEPA requested reflection without captured trajectories")

        unsupported = set(components_to_update) - {self.prompt_component}
        if unsupported:
            raise ValueError(f"unsupported prompt components: {sorted(unsupported)!r}")

        examples: list[Mapping[str, Any]] = []
        for trajectory in trajectories:
            if not isinstance(trajectory, TextTrajectory):
                raise TypeError("unexpected trajectory type")
            example = trajectory.example
            examples.append(
                {
                    "input": example.text,
                    "example_id": example.example_id,
                    "reference": example.reference,
                    "output": trajectory.output,
                    "score": trajectory.judgment.score,
                    "feedback": trajectory.judgment.feedback,
                    "objective_scores": dict(trajectory.judgment.objective_scores),
                    "metadata": dict(example.metadata),
                }
            )

        return {component: examples for component in components_to_update}


def adapter_factory(
    executor_factory: Callable[..., TaskExecutor],
    judge_factory: Callable[..., OutputJudge],
    *,
    prompt_component: str = "task_prompt",
    recoverable_exceptions: tuple[type[Exception], ...] = (),
) -> Callable[..., TextCleanupAdapter]:
    """Build a seed-aware factory for fresh, isolated GEPA adapters."""

    def create(*, seed: int | None = None) -> TextCleanupAdapter:
        return TextCleanupAdapter(
            _invoke_optional_seed_factory(executor_factory, seed),
            _invoke_optional_seed_factory(judge_factory, seed),
            prompt_component=prompt_component,
            recoverable_exceptions=recoverable_exceptions,
        )

    return create


def _invoke_optional_seed_factory(factory: Callable[..., Any], seed: int | None) -> Any:
    if seed is None:
        return factory()
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory()
    try:
        signature.bind(seed=seed)
    except TypeError:
        try:
            signature.bind(seed)
        except TypeError:
            return factory()
        return factory(seed)
    return factory(seed=seed)
