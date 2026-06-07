import os

# Set global environment overrides for tests so they are independent of local .env file
os.environ["MOCK_MODE"] = "true"
os.environ["LLM_PROVIDER"] = "gemini"
