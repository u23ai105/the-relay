"""
Batch 2: Configurable FastAPI Stand-In Provider.
Simulates OpenAI-compatible streaming endpoints with exact fault injection based
on `X-Relay-Fault` request headers for deterministic scheduler and limit testing.
"""

import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Relay Stand-In Provider")

class SimulatorState:
    in_flight: int = 0
    
state = SimulatorState()

@app.post("/openai/v1/chat/completions")
async def chat_completions(request: Request):
    req_data = await request.json()
    fault = request.headers.get("x-relay-fault", "").lower()
    
    # Static Rejections (Non-streaming errors bypass active-stream accounting)
    if fault == "429_rpm":
        return JSONResponse(
            {"error": {"message": "Rate limit exceeded (RPM)", "type": "rate_limit_error"}},
            status_code=429,
            headers={"Retry-After": "2.5"}
        )
    if fault == "429_tpd":
        return JSONResponse(
            {"error": {"message": "Tokens per day exceeded (TPD)", "type": "rate_limit_error"}},
            status_code=429
        )
    if fault == "503":
        return JSONResponse(
            {"error": {"message": "Service unavailable", "type": "server_error"}},
            status_code=503
        )
    if fault == "timeout_ttft":
        await asyncio.sleep(20.0)
        return JSONResponse({"error": "Gateway timeout"}, status_code=504)

    # SSE Streaming Responses (Yield-level concurrency accounting)
    async def event_generator():
        state.in_flight += 1
        try:
            latency = 0.01 + (state.in_flight * 0.005)
            await asyncio.sleep(latency)
            
            if fault == "disconnect_mid_stream":
                # Flush the first chunk to the network buffer, then crash the socket abruptly
                yield 'data: {"choices": [{"delta": {"content": "truncated json..."}}]}\n\n'
                
                # In FastAPI/Uvicorn, raising an exception inside the generator forces a hard socket reset (RemoteProtocolError)
                await asyncio.sleep(0.05)
                raise ConnectionError("Simulated TCP socket reset mid-stream")
                
            if fault == "hold_mid_stream":
                yield 'data: {"choices": [{"delta": {"content": "held"}}]}\n\n'
                # Hold stream open indefinitely until client disconnects or test finishes
                while True:
                    await asyncio.sleep(1.0)

            yield 'data: {"choices": [{"delta": {"content": "{"}}]}\n\n'
            await asyncio.sleep(0.01)
            yield 'data: {"choices": [{"delta": {"content": "\\"abstained\\": false,"}}]}\n\n'
            yield 'data: {"choices": [{"delta": {"content": "\\"answer_text\\": \\"Simulated valid.\\","}}]}\n\n'
            yield 'data: {"choices": [{"delta": {"content": "\\"claims\\": []}"}}]}\n\n'
            
            # Usage metadata block
            yield 'data: {"choices": [], "usage": {"prompt_tokens": 1500, "completion_tokens": 50}}\n\n'
            yield 'data: [DONE]\n\n'
        finally:
            state.in_flight -= 1

    return StreamingResponse(event_generator(), media_type="text/event-stream")
