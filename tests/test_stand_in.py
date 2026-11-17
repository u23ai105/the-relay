import asyncio
import pytest
import httpx
import uvicorn
from src.stand_in import app, state


@pytest.fixture(scope="module")
def local_test_server():
    import threading
    config = uvicorn.Config(app=app, host="127.0.0.1", port=8012, log_level="critical")
    server = uvicorn.Server(config)
    
    def run_server():
        server.run()
        
    thread = threading.Thread(target=run_server)
    thread.start()
    import time
    time.sleep(0.5) 
    yield "http://127.0.0.1:8012/openai/v1"
    server.should_exit = True
    thread.join(timeout=2.0)


@pytest.mark.asyncio
async def test_standin_valid_streaming(local_test_server):
    async with httpx.AsyncClient(base_url=local_test_server) as client:
        req = {"model": "m", "messages": [], "stream": True}
        async with client.stream("POST", "/chat/completions", json=req) as response:
            assert response.status_code == 200
            content = ""
            async for line in response.aiter_lines():
                if "answer_text" in line:
                    content += line
            assert "Simulated valid." in content


@pytest.mark.asyncio
async def test_standin_fault_429_rpm(local_test_server):
    async with httpx.AsyncClient(base_url=local_test_server) as client:
        req = {"model": "m", "messages": [], "stream": True}
        response = await client.post(
            "/chat/completions", 
            json=req, 
            headers={"x-relay-fault": "429_rpm"}
        )
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "2.5"


@pytest.mark.asyncio
async def test_standin_fault_503(local_test_server):
    async with httpx.AsyncClient(base_url=local_test_server) as client:
        req = {"model": "m", "messages": [], "stream": True}
        response = await client.post(
            "/chat/completions", 
            json=req, 
            headers={"x-relay-fault": "503"}
        )
        assert response.status_code == 503


@pytest.mark.asyncio
async def test_standin_concurrent_streaming_accounting(local_test_server):
    """
    Verifies state.in_flight correctly counts ACTIVE concurrent stream generators.
    Uses deterministic signaling to wait until the generator is definitively yielding.
    """
    async with httpx.AsyncClient(base_url=local_test_server) as client:
        req = {"model": "m", "messages": [], "stream": True}
        
        # State should be 0 before request
        assert state.in_flight == 0
        
        async with client.stream(
            "POST", 
            "/chat/completions", 
            json=req, 
            headers={"x-relay-fault": "hold_mid_stream"}
        ) as response:
            assert response.status_code == 200
            
            async for line in response.aiter_lines():
                if "held" in line:
                    break # Generator has successfully started and is now held
            
            # While actively streaming, in_flight MUST be elevated
            assert state.in_flight == 1
            
        # Exiting the async with block cancels the request locally, causing the server
        # to raise a cancellation error inside the generator.
        # Yield briefly to let the server's event loop process the disconnection and run its finally block.
        await asyncio.sleep(0.2)
        
        # Ensure it decrements properly after client cancellation
        assert state.in_flight == 0
