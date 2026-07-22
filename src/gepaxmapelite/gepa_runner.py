"""Isolated inner GEPA runs driven by an outer reflection meta-prompt.

This module deliberately has no import-time dependency on :mod:`gepa`.  The
outer search can therefore inspect, schedule, and cache individuals without
requiring GEPA (or its optional model dependencies) until an inner evaluation
actually starts.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Generic, Protocol, TypedDict, TypeVar, cast

from gepaxmapelite.models import MetaPromptGenome

DataT = TypeVar("DataT")


class DataLoaderLike(Protocol[DataT]):
    """Structural subset of GEPA's data-loader protocol used by this runner."""

    def all_ids(self) -> Sequence[Any]:
        """Return all stable example identifiers in evaluation order."""

    def fetch(self, ids: Sequence[Any]) -> list[DataT]:
        """Fetch examples for ``ids`` while preserving their order."""

    def __len__(self) -> int:
        """Return the number of available examples."""


class InnerRunResult(TypedDict):
    """JSON-serializable summary of one completed GEPA inner run."""

    best_task_candidate: dict[str, str]
    gepa_score: float
    candidate_count: int
    metric_calls: int
    run_dir: str


class GEPAInnerRunFailure(RuntimeError):
    """A completed GEPA process that cannot be treated as a valid inner evaluation."""

    def __init__(self, message: str, *, metric_calls: int, run_dir: Path) -> None:
        super().__init__(message)
        self.metric_calls = metric_calls
        self.run_dir = str(run_dir)


_CURRENT_PARAMETER_PLACEHOLDER = "<curr_param>"
_SIDE_INFORMATION_PLACEHOLDER = "<side_info>"
_REQUIRED_META_PROMPT_PLACEHOLDERS = (
    _CURRENT_PARAMETER_PLACEHOLDER,
    _SIDE_INFORMATION_PLACEHOLDER,
)


def validate_meta_prompt(meta_prompt: str) -> None:
    """Validate GEPA's public reflection-template contract.

    GEPA replaces ``<curr_param>`` with the selected task-prompt component and
    ``<side_info>`` with the adapter's reflective dataset.  Both literal tokens
    must survive every outer mutation.

    Args:
        meta_prompt: Reflection prompt template proposed by the outer search.

    Raises:
        TypeError: If ``meta_prompt`` is not a string.
        ValueError: If it is empty or omits a required placeholder.
    """

    if not isinstance(meta_prompt, str):
        raise TypeError("meta_prompt must be a string")
    if not meta_prompt.strip():
        raise ValueError("meta_prompt must not be empty")

    missing = [
        placeholder
        for placeholder in _REQUIRED_META_PROMPT_PLACEHOLDERS
        if placeholder not in meta_prompt
    ]
    if missing:
        rendered = ", ".join(missing)
        raise ValueError(f"meta_prompt is missing required placeholder(s): {rendered}")


def _copy_and_validate_task_seed(task_seed: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(task_seed, Mapping):
        raise TypeError("task_seed must be a mapping of component names to prompt text")
    if not task_seed:
        raise ValueError("task_seed must contain at least one task-prompt component")

    copied: dict[str, str] = {}
    for component, text in task_seed.items():
        if not isinstance(component, str) or not component:
            raise TypeError("every task_seed component name must be a non-empty string")
        if not isinstance(text, str):
            raise TypeError(f"task_seed[{component!r}] must be a string")
        copied[component] = text
    return copied


def _validate_data_source(data: object, name: str) -> int:
    if isinstance(data, (str, bytes)):
        raise TypeError(f"{name} must contain examples, not text bytes")
    try:
        size = len(cast(Any, data))
    except (TypeError, AttributeError) as exc:
        raise TypeError(f"{name} must be a sequence or a GEPA-compatible data loader") from exc
    if size <= 0:
        raise ValueError(f"{name} must contain at least one example")
    return size


def _validate_integer(value: int, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _load_gepa_optimize() -> Callable[..., Any]:
    """Load ``gepa.optimize`` only when an inner run is requested."""

    try:
        gepa = import_module("gepa")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "GEPA is required to execute an inner run; install the project's "
            "GEPA dependency (compatible with gepa==0.1.4)."
        ) from exc

    optimize = getattr(gepa, "optimize", None)
    if not callable(optimize):
        raise RuntimeError("the installed GEPA package does not expose callable gepa.optimize")
    return cast(Callable[..., Any], optimize)


def _allocate_unique_run_dir(root: str | Path, meta_prompt: str, seed: int) -> Path:
    """Atomically allocate a new child directory below ``root``.

    GEPA resumes whenever a run directory already contains state.  Allocating
    a fresh directory per outer individual prevents candidates, budgets, and
    evaluator state from leaking between meta-prompt fitness evaluations.
    """

    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    prompt_digest = hashlib.sha256(meta_prompt.encode("utf-8")).hexdigest()[:12]
    prefix = f"inner-s{seed}-{prompt_digest}"

    # UUID collisions are extraordinarily unlikely; the bounded loop also
    # handles a collision safely without ever reusing an existing GEPA run.
    for _ in range(100):
        candidate = root_path / f"{prefix}-{uuid.uuid4().hex[:12]}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate.resolve()
    raise RuntimeError(f"unable to allocate a unique GEPA run directory under {root_path}")


def _build_fresh_dependencies(
    adapter_factory: Callable[..., Any],
    reflection_lm_factory: Callable[..., Any],
    seed: int,
) -> tuple[Any, Any]:
    if not callable(adapter_factory):
        raise TypeError("adapter_factory must be callable")
    if not callable(reflection_lm_factory):
        raise TypeError("reflection_lm_factory must be callable")

    adapter = _invoke_seeded_factory(adapter_factory, seed, "adapter_factory")
    reflection_lm = _invoke_seeded_factory(
        reflection_lm_factory,
        seed,
        "reflection_lm_factory",
    )
    if adapter is None:
        raise TypeError("adapter_factory returned None")
    if reflection_lm is None:
        raise TypeError("reflection_lm_factory returned None")

    for method_name in ("evaluate", "make_reflective_dataset"):
        if not callable(getattr(adapter, method_name, None)):
            raise TypeError(f"adapter_factory result must implement callable {method_name}()")

    # GEPA 0.1.4 documents adapter-owned proposal generation as optional, but
    # its reflective proposer accesses this attribute directly.  Add the
    # explicit None sentinel for ordinary duck-typed adapters when possible.
    if not hasattr(adapter, "propose_new_texts"):
        try:
            adapter.propose_new_texts = None
        except (AttributeError, TypeError) as exc:
            raise TypeError(
                "adapter_factory result must expose propose_new_texts=None so "
                "GEPA can use reflection_prompt_template"
            ) from exc
    if adapter.propose_new_texts is not None:
        raise ValueError(
            "adapter.propose_new_texts must be None: an adapter-owned proposer "
            "would bypass the supplied reflection meta-prompt"
        )

    if not isinstance(reflection_lm, str) and not callable(reflection_lm):
        raise TypeError("reflection_lm_factory must return a model name or callable language model")
    return adapter, reflection_lm


class _ReflectionCallMonitor:
    """Observe provider calls that GEPA 0.1.4 may catch internally."""

    def __init__(self, delegate: Callable[..., Any]) -> None:
        self.delegate = delegate
        self.failures: list[Exception] = []
        self.successful_calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            result = self.delegate(*args, **kwargs)
        except Exception as exc:
            self.failures.append(exc)
            raise
        self.successful_calls += 1
        return result

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self.delegate, name)
        if name != "batch_complete" or not callable(attribute):
            return attribute

        def monitored_batch(*args: Any, **kwargs: Any) -> list[Any]:
            try:
                results = list(attribute(*args, **kwargs))
            except Exception as exc:
                self.failures.append(exc)
                raise
            self.successful_calls += len(results)
            return results

        return monitored_batch


class _ProposalSuccessTracker:
    """Count reflection proposals that reached GEPA's public callback boundary."""

    def __init__(self) -> None:
        self.proposal_starts = 0
        self.proposal_count = 0

    def on_proposal_start(self, _event: Any) -> None:
        self.proposal_starts += 1

    def on_proposal_end(self, _event: Any) -> None:
        self.proposal_count += 1


def _monitored_reflection_lm(reflection_lm: Any, seed: int) -> _ReflectionCallMonitor:
    """Materialize string models with a provider seed, then monitor calls."""

    delegate = reflection_lm
    if isinstance(reflection_lm, str):
        try:
            lm_module = import_module("gepa.lm")
            lm_class = lm_module.LM
            delegate = lm_class(reflection_lm, seed=seed)
        except (ImportError, AttributeError, TypeError) as exc:
            raise RuntimeError("unable to construct GEPA's seeded reflection LM") from exc
    if not callable(delegate):  # guarded earlier; keeps this boundary explicit
        raise TypeError("reflection language model must be callable")
    return _ReflectionCallMonitor(delegate)


def _invoke_seeded_factory(factory: Callable[..., Any], seed: int, name: str) -> Any:
    """Pass the replicate seed when a dependency factory supports it."""

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
            try:
                signature.bind()
            except TypeError as exc:
                raise TypeError(f"{name} must accept seed or no arguments") from exc
            return factory()
        return factory(seed)
    return factory(seed=seed)


def _summarize_gepa_result(result: Any, run_dir: Path) -> InnerRunResult:
    try:
        best_idx = int(result.best_idx)
        aggregate_scores = result.val_aggregate_scores
        raw_best_candidate = result.best_candidate
        candidate_count = int(result.num_candidates)
        raw_metric_calls = result.total_metric_calls
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("gepa.optimize returned an incompatible result object") from exc

    if best_idx < 0 or best_idx >= len(aggregate_scores):
        raise RuntimeError("GEPA returned an invalid best candidate index")
    gepa_score = float(aggregate_scores[best_idx])
    if not math.isfinite(gepa_score):
        raise RuntimeError("GEPA returned a non-finite best validation score")
    if candidate_count < 1:
        raise RuntimeError("GEPA returned no task-prompt candidates")
    if raw_metric_calls is None:
        raise RuntimeError("GEPA did not report total metric calls")
    metric_calls = int(raw_metric_calls)
    if metric_calls < 0:
        raise RuntimeError("GEPA returned a negative metric-call count")

    if not isinstance(raw_best_candidate, Mapping):
        raise RuntimeError("GEPA returned a non-mapping task candidate for a mapping seed")
    best_task_candidate: dict[str, str] = {}
    for component, text in raw_best_candidate.items():
        if not isinstance(component, str) or not isinstance(text, str):
            raise RuntimeError(
                "GEPA returned a task candidate that is not JSON-safe dict[str, str]"
            )
        best_task_candidate[component] = text

    return InnerRunResult(
        best_task_candidate=best_task_candidate,
        gepa_score=gepa_score,
        candidate_count=candidate_count,
        metric_calls=metric_calls,
        run_dir=str(run_dir),
    )


def run_gepa_inner(
    *,
    meta_prompt: str,
    task_seed: Mapping[str, str],
    trainset: Sequence[DataT] | DataLoaderLike[DataT],
    valset: Sequence[DataT] | DataLoaderLike[DataT],
    adapter_factory: Callable[..., Any],
    reflection_lm_factory: Callable[..., Any],
    run_dir: str | Path,
    seed: int = 0,
    max_metric_calls: int,
) -> InnerRunResult:
    """Evaluate one outer meta-prompt through an isolated inner GEPA run.

    ``meta_prompt`` remains fixed for this run and controls GEPA's reflective
    mutation of the task prompts in ``task_seed``.  The supplied factories are
    invoked exactly once per call so model counters, adapter caches, and other
    mutable state cannot leak across outer individuals.  ``run_dir`` is a root;
    the function atomically creates and passes a unique child directory to
    GEPA, preventing its automatic resume behavior from mixing experiments.

    The GEPA metric-call stopper is checked between iterations, so GEPA may
    exceed ``max_metric_calls`` by the cost of the final iteration.  The actual
    count is always returned in ``InnerRunResult.metric_calls``.

    Args:
        meta_prompt: Outer individual's reflection template.  It must contain
            the literal ``<curr_param>`` and ``<side_info>`` placeholders.
        task_seed: Fixed inner candidate mapping component names to task-prompt
            text.  A defensive copy is passed to GEPA.
        trainset: Inner reflective-training examples or a GEPA-compatible
            loader.
        valset: Inner validation examples or a GEPA-compatible loader.
        adapter_factory: Factory for a fresh GEPA adapter. If it accepts
            ``seed`` (keyword or positional), it receives the replicate seed.
            Its ``propose_new_texts`` attribute must be ``None`` so the
            meta-prompt controls proposal generation.
        reflection_lm_factory: Factory returning a fresh GEPA reflection-LM
            callable or model-name string. Seed-aware factories receive the
            replicate seed. Model-name strings are materialized with GEPA's LM
            wrapper and receive the same provider sampling seed.
        run_dir: Parent directory in which an isolated child run is allocated.
        seed: GEPA's sampling/random seed.
        max_metric_calls: Inner evaluation budget. It must exceed the valset
            size so at least one reflection proposal can use the meta-prompt.

    Returns:
        A JSON-safe dictionary with the best evolved task candidate, its GEPA
        validation score, candidate and metric-call counts, and actual run
        directory.

    Raises:
        TypeError: If an input or factory product violates its contract.
        ValueError: If the template, seed mapping, datasets, or budget is
            invalid.
        RuntimeError: If GEPA is unavailable or returns an incompatible result.
        Exception: Evaluator and GEPA optimization errors are propagated. A
            reflection-provider error that survives GEPA's retry path, or a run
            that never produces a valid reflection proposal, is converted to a
            failed individual instead of a valid-looking seed-only result.
    """

    validate_meta_prompt(meta_prompt)
    copied_task_seed = _copy_and_validate_task_seed(task_seed)
    _validate_data_source(trainset, "trainset")
    valset_size = _validate_data_source(valset, "valset")
    _validate_integer(seed, "seed", minimum=0)
    _validate_integer(max_metric_calls, "max_metric_calls", minimum=1)
    if max_metric_calls <= valset_size:
        raise ValueError(
            "max_metric_calls must exceed the valset size so GEPA performs at least "
            "one reflection step with the supplied meta-prompt"
        )

    optimize = _load_gepa_optimize()
    adapter, reflection_lm = _build_fresh_dependencies(
        adapter_factory,
        reflection_lm_factory,
        seed,
    )
    monitored_lm = _monitored_reflection_lm(reflection_lm, seed)
    proposal_tracker = _ProposalSuccessTracker()
    isolated_run_dir = _allocate_unique_run_dir(run_dir, meta_prompt, seed)

    result = optimize(
        seed_candidate=copied_task_seed,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=monitored_lm,
        reflection_prompt_template=meta_prompt,
        module_selector="round_robin",
        use_merge=False,
        max_metric_calls=max_metric_calls,
        run_dir=str(isolated_run_dir),
        track_best_outputs=False,
        display_progress_bar=False,
        cache_evaluation=False,
        seed=seed,
        raise_on_exception=True,
        val_evaluation_policy="full_eval",
        skip_perfect_score=False,
        callbacks=[proposal_tracker],
    )
    summary = _summarize_gepa_result(result, isolated_run_dir)
    if proposal_tracker.proposal_starts > proposal_tracker.proposal_count:
        if monitored_lm.failures:
            raise GEPAInnerRunFailure(
                "GEPA reflection model failed before completing every started proposal",
                metric_calls=summary["metric_calls"],
                run_dir=isolated_run_dir,
            ) from monitored_lm.failures[-1]
        raise GEPAInnerRunFailure(
            "GEPA did not complete every started reflection proposal; the inner run is incomplete",
            metric_calls=summary["metric_calls"],
            run_dir=isolated_run_dir,
        )
    if proposal_tracker.proposal_count < 1:
        if monitored_lm.failures:
            raise GEPAInnerRunFailure(
                "GEPA reflection model failed before producing a valid proposal",
                metric_calls=summary["metric_calls"],
                run_dir=isolated_run_dir,
            ) from monitored_lm.failures[-1]
        raise GEPAInnerRunFailure(
            "GEPA completed without a valid reflection proposal; the meta-prompt "
            "was not meaningfully evaluated",
            metric_calls=summary["metric_calls"],
            run_dir=isolated_run_dir,
        )
    return summary


@dataclass(frozen=True, slots=True)
class GEPAInnerRunner(Generic[DataT]):
    """Configured adapter from outer genomes to isolated GEPA inner runs.

    The dataset and downstream seed remain fixed across outer individuals.
    Fresh adapter and reflection-LM factories are invoked for every run.
    Sequences are copied. Custom mutable data loaders require corresponding
    factories so loader state cannot leak between meta-prompts.
    """

    task_seed: Mapping[str, str]
    trainset: Sequence[DataT] | DataLoaderLike[DataT]
    valset: Sequence[DataT] | DataLoaderLike[DataT]
    adapter_factory: Callable[..., Any]
    reflection_lm_factory: Callable[..., Any]
    max_metric_calls: int
    trainset_factory: Callable[[], Sequence[DataT] | DataLoaderLike[DataT]] | None = None
    valset_factory: Callable[[], Sequence[DataT] | DataLoaderLike[DataT]] | None = None

    def run(
        self,
        genome: MetaPromptGenome,
        *,
        replicate_seed: int,
        run_dir: str | Path,
    ) -> InnerRunResult:
        if not isinstance(genome, MetaPromptGenome):
            raise TypeError("genome must be a MetaPromptGenome")
        trainset = _fresh_data_source(self.trainset, self.trainset_factory, "trainset")
        valset = _fresh_data_source(self.valset, self.valset_factory, "valset")
        return run_gepa_inner(
            meta_prompt=genome.reflection_template,
            task_seed=self.task_seed,
            trainset=trainset,
            valset=valset,
            adapter_factory=self.adapter_factory,
            reflection_lm_factory=self.reflection_lm_factory,
            run_dir=run_dir,
            seed=replicate_seed,
            max_metric_calls=self.max_metric_calls,
        )


def _fresh_data_source(
    source: Sequence[DataT] | DataLoaderLike[DataT],
    factory: Callable[[], Sequence[DataT] | DataLoaderLike[DataT]] | None,
    name: str,
) -> Sequence[DataT] | DataLoaderLike[DataT]:
    if factory is not None:
        fresh = factory()
        _validate_data_source(fresh, name)
        return fresh
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        return list(source)
    raise TypeError(
        f"GEPAInnerRunner requires {name}_factory when {name} is a custom DataLoader; "
        "reusing a mutable loader would leak state across meta-prompts"
    )


__all__ = [
    "DataLoaderLike",
    "GEPAInnerRunFailure",
    "GEPAInnerRunner",
    "InnerRunResult",
    "run_gepa_inner",
    "validate_meta_prompt",
]
