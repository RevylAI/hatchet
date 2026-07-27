import asyncio
from types import SimpleNamespace
from typing import Any, cast

from hatchet_sdk.runnables.action import Action, ActionPayload, ActionType
from hatchet_sdk.worker.runner.runner import Runner


class FakeContext:
    exit_flag = False

    def __init__(self) -> None:
        self.is_cancelled = False

    def _set_cancellation_flag(self) -> None:
        self.is_cancelled = True


class FakeRunner:
    def __init__(
        self,
        tasks: dict[str, asyncio.Task[Any]],
        contexts: dict[str, FakeContext],
    ) -> None:
        self.tasks = tasks
        self.contexts = contexts
        self.threads: dict[str, object] = {}
        self.cancellations: dict[str, bool] = {}
        self.config = SimpleNamespace(enable_force_kill_sync_threads=False)
        self.cleaned_keys: list[str] = []

    def cleanup_run_id(self, key: str) -> None:
        self.cleaned_keys.append(key)


def make_action(
    *,
    step_run_id: str,
    action_type: ActionType,
    invocation_count: int | None,
    retry_count: int = 0,
) -> Action:
    return Action(
        worker_id="worker",
        tenant_id="tenant",
        workflow_run_id="workflow",
        job_id="job",
        job_name="job",
        job_run_id="job-run",
        step_id="step",
        step_run_id=step_run_id,
        action_id="execute_and_finalize",
        action_type=action_type,
        retry_count=retry_count,
        action_payload=ActionPayload(),
        durable_task_invocation_count=invocation_count,
    )


async def cancel(runner: FakeRunner, action: Action) -> None:
    await Runner.handle_cancel_action(cast(Runner, runner), action)
    await asyncio.sleep(0)


async def cleanup(tasks: dict[str, asyncio.Task[Any]]) -> None:
    for task in tasks.values():
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks.values(), return_exceptions=True)


async def test_countless_cancel_reaches_durable_task() -> None:
    start = make_action(
        step_run_id="step",
        action_type=ActionType.START_STEP_RUN,
        invocation_count=1,
    )
    action = make_action(
        step_run_id="step",
        action_type=ActionType.CANCEL_STEP_RUN,
        invocation_count=None,
    )
    gate = asyncio.Event()
    tasks = {start.key: asyncio.create_task(gate.wait())}
    contexts = {start.key: FakeContext()}
    runner = FakeRunner(tasks, contexts)

    try:
        await cancel(runner, action)

        assert action.key == "step/0"
        assert start.key == "step/0/1"
        assert contexts[start.key].is_cancelled
        assert tasks[start.key].cancelled()
        assert runner.cancellations == {start.key: True}
        assert runner.cleaned_keys == [start.key]
    finally:
        await cleanup(tasks)


async def test_non_durable_exact_match_still_cancelled() -> None:
    start = make_action(
        step_run_id="step",
        action_type=ActionType.START_STEP_RUN,
        invocation_count=None,
    )
    action = make_action(
        step_run_id="step",
        action_type=ActionType.CANCEL_STEP_RUN,
        invocation_count=None,
    )
    gate = asyncio.Event()
    tasks = {start.key: asyncio.create_task(gate.wait())}
    contexts = {start.key: FakeContext()}
    runner = FakeRunner(tasks, contexts)

    try:
        await cancel(runner, action)

        assert contexts[start.key].is_cancelled
        assert tasks[start.key].cancelled()
        assert runner.cleaned_keys == [start.key]
    finally:
        await cleanup(tasks)


async def test_prefix_similar_step_id_and_other_retry_are_isolated() -> None:
    target = make_action(
        step_run_id="step-a",
        action_type=ActionType.START_STEP_RUN,
        invocation_count=1,
    )
    prefix_similar = make_action(
        step_run_id="step-ab",
        action_type=ActionType.START_STEP_RUN,
        invocation_count=1,
    )
    other_retry = make_action(
        step_run_id="step-a",
        action_type=ActionType.START_STEP_RUN,
        invocation_count=1,
        retry_count=1,
    )
    action = make_action(
        step_run_id="step-a",
        action_type=ActionType.CANCEL_STEP_RUN,
        invocation_count=None,
    )
    gate = asyncio.Event()
    tasks = {
        target.key: asyncio.create_task(gate.wait()),
        prefix_similar.key: asyncio.create_task(gate.wait()),
        other_retry.key: asyncio.create_task(gate.wait()),
    }
    contexts = {key: FakeContext() for key in tasks}
    runner = FakeRunner(tasks, contexts)

    try:
        await cancel(runner, action)

        assert contexts[target.key].is_cancelled
        assert tasks[target.key].cancelled()
        assert not contexts[prefix_similar.key].is_cancelled
        assert not tasks[prefix_similar.key].cancelled()
        assert not contexts[other_retry.key].is_cancelled
        assert not tasks[other_retry.key].cancelled()
        assert runner.cleaned_keys == [target.key]
    finally:
        await cleanup(tasks)


async def test_no_registered_task_falls_back_to_action_key() -> None:
    action = make_action(
        step_run_id="ghost-step",
        action_type=ActionType.CANCEL_STEP_RUN,
        invocation_count=None,
    )
    runner = FakeRunner(tasks={}, contexts={})

    await cancel(runner, action)

    assert runner.cleaned_keys == [action.key]
    assert runner.cancellations == {}


async def test_countless_cancel_reaches_all_durable_incarnations() -> None:
    starts = [
        make_action(
            step_run_id="step",
            action_type=ActionType.START_STEP_RUN,
            invocation_count=invocation_count,
        )
        for invocation_count in (1, 2)
    ]
    action = make_action(
        step_run_id="step",
        action_type=ActionType.CANCEL_STEP_RUN,
        invocation_count=None,
    )
    gate = asyncio.Event()
    tasks = {start.key: asyncio.create_task(gate.wait()) for start in starts}
    contexts = {key: FakeContext() for key in tasks}
    runner = FakeRunner(tasks, contexts)

    try:
        await cancel(runner, action)

        assert all(context.is_cancelled for context in contexts.values())
        assert all(task.cancelled() for task in tasks.values())
        assert runner.cleaned_keys == ["step/0/1", "step/0/2"]
    finally:
        await cleanup(tasks)


async def test_cancel_with_invocation_suffix_reaches_all_incarnations() -> None:
    starts = [
        make_action(
            step_run_id="step",
            action_type=ActionType.START_STEP_RUN,
            invocation_count=invocation_count,
        )
        for invocation_count in (1, 2)
    ]
    action = make_action(
        step_run_id="step",
        action_type=ActionType.CANCEL_STEP_RUN,
        invocation_count=2,
    )
    gate = asyncio.Event()
    tasks = {start.key: asyncio.create_task(gate.wait()) for start in starts}
    contexts = {key: FakeContext() for key in tasks}
    runner = FakeRunner(tasks, contexts)

    try:
        await cancel(runner, action)

        assert action.key == "step/0/2"
        assert all(context.is_cancelled for context in contexts.values())
        assert all(task.cancelled() for task in tasks.values())
        assert runner.cleaned_keys == ["step/0/1", "step/0/2"]
    finally:
        await cleanup(tasks)
