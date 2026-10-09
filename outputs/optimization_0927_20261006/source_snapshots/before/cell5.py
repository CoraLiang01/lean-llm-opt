class ModelOutputTruncated(RuntimeError):
    """The API explicitly reported an incomplete model response."""


class RejectTruncatedOutput(BaseCallbackHandler):
    raise_error = True

    def on_llm_end(self, response, **kwargs):
        for choices in response.generations:
            for generation in choices:
                metadata = getattr(getattr(generation, "message", None), "response_metadata", {}) or {}
                info = generation.generation_info or {}
                reason = str(info.get("finish_reason") or metadata.get("finish_reason") or "").lower()
                if reason in {"length", "max_tokens", "max_output_tokens"}:
                    error = ModelOutputTruncated(f"Model output is incomplete: finish_reason={reason}")
                    error.llm_usage = (response.llm_output or {}).get("token_usage", {})
                    raise error


def require_api_key():
    """Load credentials at use time, including after configuring a running kernel."""
    load_dotenv(PROJECT_ROOT / ".env.local", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    key = (os.environ.get("OPENAI_API_KEY") or user_api_key or "").strip()
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Set it in the environment or in "
            f"{PROJECT_ROOT / '.env.local'} before running a test."
        )
    return key


def make_llm(*, timeout=180):
    return ChatOpenAI(
        model_name=MODEL_SNAPSHOT, openai_api_key=require_api_key(),
        temperature=0.0, top_p=1, n=1, max_retries=0, timeout=timeout,
        callbacks=[RejectTruncatedOutput()],
    )


def make_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=require_api_key(), max_retries=0)
