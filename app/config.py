import os

from dotenv import load_dotenv

REQUIRED_KEYS = ("GROQ_API_KEY", "TAVILY_API_KEY")


def validate_env():
    load_dotenv()
    missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )
