import os
import asyncio
from google import genai
from google.genai import types

async def probe():
    api_key = os.environ.get("GEMINI_API_KEY")
    # Must use v1alpha for the bidirectional WebSocket endpoint
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    
    models_to_test = [
        "gemini-2.0-flash-exp",
        "gemini-2.5-flash-preview-native-audio-dialog",
        "gemini-live-2.5-flash-preview"
    ]
    
    config = types.LiveConnectConfig(response_modalities=["AUDIO"])

    for model in models_to_test:
        print(f"Testing {model}...")
        try:
            async with client.aio.live.connect(model=model, config=config) as session:
                print(f"[SUCCESS] Connected successfully to {model}!")
                return model
        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == "__main__":
    asyncio.run(probe())