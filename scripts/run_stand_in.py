"""
Batch 2: Development server runner for the Stand-In Provider.
"""
import uvicorn

if __name__ == "__main__":
    print("Starting Relay Stand-In Provider on http://127.0.0.1:8001")
    uvicorn.run("src.stand_in:app", host="127.0.0.1", port=8001, log_level="info")
