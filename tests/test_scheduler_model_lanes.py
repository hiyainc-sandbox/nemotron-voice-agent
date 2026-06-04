import asyncio

import pytest

from nemotron_speech.server import ASRServer, ASRSession, batch_group_key


LANE_ENV = (
    "NEMOTRON_CONTINUOUS",
    "NEMOTRON_SCHEDULER_B1",
    "NEMOTRON_BATCH_SCHED",
    "NEMOTRON_MODEL_LANES",
)


@pytest.fixture(autouse=True)
def clean_lane_env(monkeypatch):
    for name in LANE_ENV:
        monkeypatch.delenv(name, raising=False)


def make_server(monkeypatch, **env):
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))

    import nemotron_speech.server as server_mod

    monkeypatch.setattr(server_mod.torch.cuda, "is_available", lambda: True)
    return ASRServer("dummy-model")


def test_model_lanes_default_off_and_scheduler_fallback(monkeypatch):
    server = make_server(monkeypatch)
    assert server.model_lanes_requested == 1
    assert server.model_lanes == 1

    server = make_server(monkeypatch, NEMOTRON_MODEL_LANES=2)
    assert server.model_lanes_requested == 2
    assert server.model_lanes == 1

    server = make_server(
        monkeypatch,
        NEMOTRON_CONTINUOUS=1,
        NEMOTRON_SCHEDULER_B1=1,
        NEMOTRON_BATCH_SCHED=1,
        NEMOTRON_MODEL_LANES=2,
    )
    assert server.model_lanes_requested == 2
    assert server.model_lanes == 2


def test_parallel_lanes_only_accept_steady_same_drop_extra(monkeypatch):
    server = make_server(
        monkeypatch,
        NEMOTRON_CONTINUOUS=1,
        NEMOTRON_SCHEDULER_B1=1,
        NEMOTRON_BATCH_SCHED=1,
        NEMOTRON_MODEL_LANES=2,
    )
    server.shift_frames = 16
    server.pre_encode_cache_size = 9
    server.drop_extra = 2
    server.prompted_model = False

    steady_key = (*batch_group_key("en-US", False, 2, 25, "greedy_batch"), False, False)
    first_key = (*batch_group_key("en-US", False, 0, 16, "greedy_batch"), True, True)

    assert server._scheduler_batch_key_parallel_lane_key(steady_key) == (None, 2)
    assert server._scheduler_batch_key_parallel_lane_key(first_key) is None

    assert server._scheduler_reserve_model_lane_for_key(steady_key) == (0, False)
    assert server._scheduler_model_lane_key_can_dispatch(steady_key)

    assert server._scheduler_reserve_model_lane_for_key(steady_key) == (1, False)
    assert not server._scheduler_model_lane_key_can_dispatch(steady_key)
    assert not server._scheduler_model_lane_key_can_dispatch(first_key)

    server._scheduler_available_model_lanes = {0, 1}
    server._scheduler_model_lane_active_key = None
    assert server._scheduler_reserve_model_lane_for_key(
        first_key,
        preferred_lane=1,
    ) == (1, True)


def test_inflight_session_is_held_out_of_ready_set(monkeypatch):
    server = make_server(
        monkeypatch,
        NEMOTRON_CONTINUOUS=1,
        NEMOTRON_SCHEDULER_B1=1,
        NEMOTRON_BATCH_SCHED=1,
        NEMOTRON_MODEL_LANES=2,
    )
    session = ASRSession(id="s1", websocket=None)
    server._scheduler_session_ready = lambda _session: True

    server._scheduler_inflight_sessions.add(session.id)
    server._scheduler_ready.add(session.id)
    server._scheduler_mark_ready_if_ready_locked(session)

    assert session.id not in server._scheduler_ready


def test_parallel_lanes_pin_sessions_to_preferred_lane(monkeypatch):
    server = make_server(
        monkeypatch,
        NEMOTRON_CONTINUOUS=1,
        NEMOTRON_SCHEDULER_B1=1,
        NEMOTRON_BATCH_SCHED=1,
        NEMOTRON_MODEL_LANES=2,
    )
    server.shift_frames = 16
    server.pre_encode_cache_size = 9
    server.drop_extra = 2
    server.prompted_model = False

    steady_key = (*batch_group_key("en-US", False, 2, 25, "greedy_batch"), False, False)
    sessions = [
        ASRSession(id="s0", websocket=None),
        ASRSession(id="s1", websocket=None),
        ASRSession(id="s2", websocket=None),
    ]
    server._scheduler_session_model_lane_affinity["s1"] = 1

    server._scheduler_available_model_lanes = {0, 1}
    selected = server._scheduler_select_lane_affine_sessions(steady_key, sessions)
    assert [session.id for session in selected] == ["s1", "s0", "s2"]
    assert server._scheduler_preferred_lane_for_sessions(selected) == 1
    assert server._scheduler_reserve_model_lane_for_key(
        steady_key,
        preferred_lane=1,
    ) == (1, False)

    server._scheduler_available_model_lanes = {0}
    selected = server._scheduler_select_lane_affine_sessions(steady_key, sessions)
    assert [session.id for session in selected] == ["s0", "s2"]
    assert not server._scheduler_session_affinity_allows_dispatch(
        sessions[1],
        steady_key,
    )


def test_exclusive_batches_keep_session_lane_affinity(monkeypatch):
    server = make_server(
        monkeypatch,
        NEMOTRON_CONTINUOUS=1,
        NEMOTRON_SCHEDULER_B1=1,
        NEMOTRON_BATCH_SCHED=1,
        NEMOTRON_MODEL_LANES=2,
    )
    server.shift_frames = 16
    server.pre_encode_cache_size = 9
    server.drop_extra = 2
    first_key = (*batch_group_key("en-US", False, 0, 16, "greedy_batch"), True, True)
    sessions = [
        ASRSession(id="s0", websocket=None),
        ASRSession(id="s1", websocket=None),
    ]
    server._scheduler_session_model_lane_affinity["s0"] = 1
    server._scheduler_session_model_lane_affinity["s1"] = 0

    server._scheduler_available_model_lanes = {0, 1}
    selected = server._scheduler_select_lane_affine_sessions(first_key, sessions)
    assert [session.id for session in selected] == ["s1"]
    assert server._scheduler_reserve_model_lane_for_key(
        first_key,
        preferred_lane=0,
    ) == (0, True)

    server._scheduler_available_model_lanes = {1}
    assert not server._scheduler_session_affinity_allows_dispatch(
        sessions[1],
        first_key,
    )


def test_inflight_queued_events_do_not_spin_scheduler(monkeypatch):
    server = make_server(
        monkeypatch,
        NEMOTRON_CONTINUOUS=1,
        NEMOTRON_SCHEDULER_B1=1,
        NEMOTRON_BATCH_SCHED=1,
        NEMOTRON_MODEL_LANES=2,
    )
    session = ASRSession(id="s1", websocket=None)
    session.continuous_event_queue = asyncio.Queue()
    session.continuous_event_queue.put_nowait(("vad_stop",))
    server.sessions[session.id] = session

    server._scheduler_inflight_sessions.add(session.id)
    assert not server._scheduler_has_queued_events()
    assert not server._scheduler_has_work_or_due_timer()

    server._scheduler_inflight_sessions.clear()
    assert server._scheduler_has_queued_events()
    assert server._scheduler_has_work_or_due_timer()
