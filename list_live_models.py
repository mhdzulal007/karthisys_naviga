import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("\n--- Models Supporting bidiGenerateContent ---")
found = False
for model in client.models.list():
    supported = getattr(model, "supported_actions", None) or getattr(model, "supported_generation_methods", [])
    if any("bidi" in str(method).lower() for method in supported):
        print(f"-> {model.name}")
        found = True

if not found:
    print("No models with bidiGenerateContent found. Printing all flash/experimental models:")
    for model in client.models.list():
        if any(term in model.name.lower() for term in ["flash", "exp", "2.0", "live"]):
            print(f"-> {model.name}")