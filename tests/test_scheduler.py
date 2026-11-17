import asyncio
import time
import pytest
import uvicorn
import httpx

from src.scheduler import RequestContext, ProviderScheduler
from src.groq_client import GroqCompatibleClient
from src.stand_in import app, state


@pytest.fixture
def stand_in_server():
    """Starts a real local Uvicorn TCP server for authentic network-drop testing."""
    import threading
    config = uvicorn.Config(app=app, host="127.0.0.1", port=8011, log_level="critical")
    server = uvicorn.Server(config)
    
    def run_server():
        server.run()
        
    thread = threading.Thread(target=run_server)
    thread.start()
    time.sleep(0.2) # Allow startup
    yield "http://127.0.0.1:8011/openai/v1"
    server.should_exit = True
    thread.join(timeout=2.0)


@pytest.fixture
def mock_client(stand_in_server):
    config = {
        "provider": {"standin_base_url": stand_in_server, "max_transport_retries": 1},
        "pipeline": {"retry_strategy": "naive"}
    }
    return GroqCompatibleClient(config, source="stand-in", http_client=None)


@pytest.mark.asyncio
async def test_midstream_disconnect_retry(mock_client):
    config = {
        "pipeline": {"retry_strategy": "naive"},
        "provider": {"max_transport_retries": 1, "retry_base_delay_ms": 10, "retry_max_delay_ms": 10}
    }
    sched = ProviderScheduler(config, mock_client)
    ctx = RequestContext("r1", time.perf_counter(), time.perf_counter() + 10.0)
    
    # Hits the Uvicorn server, which flushes SSE then triggers ConnectionError resulting in RemoteProtocolError
    res, _ = await sched.execute_call(
        ctx, [{"role": "user", "content": "hi"}], "generate", "m1", 50, 0.0, fault_inject="disconnect_mid_stream"
    )
    assert res.success is False
    # Incomplete provider output is deliberately discarded after a mid-stream disconnect.
    assert res.content == ""
    assert res.failure_reason == "disconnect_mid_stream"
    assert res.accounting_mode == "partial_estimated"
    assert res.tokens_out > 0
    assert ctx.transport_retries == 1


@pytest.mark.asyncio
async def test_admission_and_overload(mock_client):
    config = {
        "pipeline": {"max_local_queue_depth": 2, "max_local_concurrency": 1},
        "provider": {"max_provider_concurrency": 10}
    }
    sched = ProviderScheduler(config, mock_client)
    
    ctx1 = RequestContext("r1", time.perf_counter(), time.perf_counter() + 10.0)
    ctx2 = RequestContext("r2", time.perf_counter(), time.perf_counter() + 10.0)
    ctx3 = RequestContext("r3", time.perf_counter(), time.perf_counter() + 10.0)
    ctx4 = RequestContext("r4", time.perf_counter(), time.perf_counter() + 10.0)
    
    # R1 admitted immediately
    admit1 = await sched.acquire_admission(ctx1)
    assert admit1 is True
    
    # R2 queues (depth = 1)
    t2 = asyncio.create_task(sched.acquire_admission(ctx2))
    await asyncio.sleep(0.05)
    
    # R3 queues (depth = 2)
    t3 = asyncio.create_task(sched.acquire_admission(ctx3))
    await asyncio.sleep(0.05)
    
    # R4 strictly rejected because queue depth == max_local_queue_depth
    admit4 = await sched.acquire_admission(ctx4)
    assert admit4 is False
    
    sched.release_admission(ctx1)
    
    # Release tasks to complete cleanly
    assert await t2 is True
    sched.release_admission(ctx2)
    assert await t3 is True
    sched.release_admission(ctx3)


@pytest.mark.asyncio
async def test_shared_cooldown_and_concurrency_release(mock_client):
    config = {
        "pipeline": {"retry_strategy": "aware", "max_local_concurrency": 10},
        "provider": {"max_provider_concurrency": 5, "retry_max_delay_ms": 100}
    }
    sched = ProviderScheduler(config, mock_client)
    ctx = RequestContext("r1", time.perf_counter(), time.perf_counter() + 10.0)
    
    call_task = asyncio.create_task(
        sched.execute_call(ctx, [{"role": "user", "content": "hi"}], "generate", "m1", 50, 0.0, fault_inject="429_rpm")
    )
    
    await asyncio.sleep(0.1)
    # The 429 response clears the cooldown barrier
    assert not sched.shared_cooldown.is_set()
    # Concurrency MUST be released during the backoff sleep
    assert sched.in_flight == 0
    call_task.cancel()


@pytest.mark.asyncio
async def test_daily_exhaustion_no_retry(mock_client):
    config = {
        "pipeline": {"retry_strategy": "aware"},
        "provider": {}
    }
    sched = ProviderScheduler(config, mock_client)
    ctx = RequestContext("r1", time.perf_counter(), time.perf_counter() + 10.0)
    
    res, _ = await sched.execute_call(
        ctx, [{"role": "user", "content": "hi"}], "generate", "m1", 50, 0.0, fault_inject="429_tpd"
    )
    assert res.success is False
    assert res.failure_reason == "tpd_limit"
    assert ctx.transport_retries == 0


@pytest.mark.asyncio
async def test_cancellation_safety(mock_client):
    config = {
        "pipeline": {"retry_strategy": "naive"},
        "provider": {"max_provider_concurrency": 1}
    }
    sched = ProviderScheduler(config, mock_client)
    ctx = RequestContext("r1", time.perf_counter(), time.perf_counter() + 10.0)
    
    t1 = asyncio.create_task(
        sched.execute_call(ctx, [{"role": "user", "content": "hi"}], "generate", "m1", 50, 0.0, fault_inject="timeout_ttft")
    )
    await asyncio.sleep(0.1)
    assert sched.in_flight == 1
    t1.cancel()
    
    try:
        await t1
    except asyncio.CancelledError:
        pass
        
    assert sched.in_flight == 0
    
    ctx2 = RequestContext("r2", time.perf_counter(), time.perf_counter() + 10.0)
    res, _ = await sched.execute_call(ctx2, [{"role": "user", "content": "hi"}], "generate", "m1", 50, 0.0)
    assert res.success is True


@pytest.mark.asyncio
async def test_quota_reservation_exhaustion(mock_client):
    config = {
        "pipeline": {"retry_strategy": "naive"},
        "provider": {"limit_rpm": 1}
    }
    sched = ProviderScheduler(config, mock_client)
    ctx1 = RequestContext("r1", time.perf_counter(), time.perf_counter() + 10.0)
    ctx2 = RequestContext("r2", time.perf_counter(), time.perf_counter() + 10.0)
    
    # First consumes 1 RPM limit
    await sched.execute_call(ctx1, [{"role": "user", "content": "hi"}], "generate", "m1", 50, 0.0)
    
    # Second should fail local quota
    res2, _ = await sched.execute_call(ctx2, [{"role": "user", "content": "hi"}], "generate", "m1", 50, 0.0)
    
    assert res2.success is False
    assert res2.failure_reason == "rpm_limit"
    assert res2.error_code == "local_quota"
