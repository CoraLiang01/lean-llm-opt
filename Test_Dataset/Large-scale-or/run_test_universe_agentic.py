import os
import time
import re
import ast
import math
import subprocess
import argparse
import sys
import tempfile
import pandas as pd
from dotenv import load_dotenv

# Load .env file
load_dotenv()


# Air-NRM large-scale uses one query CSV without a Dataset_address column.
# All 15 instances share the same five raw data files.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
AIR_NRM_LS_DIR = os.path.join(REPO_ROOT, "Test_Dataset", "Air_NRM", "large_scale")
AIR_NRM_QUERY_CSV = os.path.join(AIR_NRM_LS_DIR, "query_CA.csv")
AIR_NRM_DATA_FILES = [
    os.path.join(AIR_NRM_LS_DIR, filename)
    for filename in (
        "flight.csv",
        "od_demand.csv",
        "v1.csv",
        "v2.csv",
        "flight_capacity.csv",
    )
]

RESPONSE_FORMAT_REQUIREMENTS = (
    "Your final response must contain exactly these two sections:\n\n"
    "### Mathematical Model\n"
    "A concise mathematical formulation.\n\n"
    "### Python Code\n"
    "Exactly one fenced Python code block in this form:\n"
    "```python\n"
    "# Complete, directly executable gurobipy code\n"
    "```\n\n"
    "The code must be complete rather than pseudocode, must import gurobipy, "
    "create a gurobipy Model, call optimize(), and leave Gurobi output enabled "
    "so the optimal objective can be parsed. Do not put exploratory code or a "
    "second Python code block in the final response."
)

SYSTEM_INSTRUCTIONS = (
    "You are an expert in operations research and mathematical optimization. "
    "Use the provided raw data files to formulate the requested optimization "
    "problem and provide complete gurobipy code.\n\n"
    + RESPONSE_FORMAT_REQUIREMENTS
)

GEMINI_SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS


# ==========================================
# 1. Argument Parsing
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="LLM4OR Benchmark - Universal Runner")

    parser.add_argument("--model_type", type=str, default=None,
                        choices=["gemini", "openai", "claude"],
                        help="LLM provider: gemini, openai, or claude")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Specific model name (overrides default)")
    parser.add_argument("--if_proxy", action="store_true", default=None,
                        help="Enable proxy")
    parser.add_argument("--no_proxy", action="store_true",
                        help="Disable proxy")
    parser.add_argument("--proxy_url", type=str, default=None,
                        help="Proxy URL")
    parser.add_argument("--input_csv", type=str, default=None,
                        help="Input CSV path")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="Rows per batch")
    parser.add_argument("--skip_rows", type=int, default=None,
                        help="Number of rows to skip from the beginning")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Code execution timeout in seconds")
    parser.add_argument("--max_retries", type=int, default=None,
                        help="Max retries for API calls / file uploads")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Maximum output-token budget for one model response")
    parser.add_argument("--data_base_dir", type=str, default=None,
                        help="Base directory prepended to Dataset_address paths in CSV")

    return parser.parse_args()


def get_config(args):
    """Merge .env defaults with CLI overrides. CLI args take priority."""
    config = {}
    config["model_type"] = args.model_type or os.getenv("MODEL_TYPE", "gemini")
    config["proxy_url"] = args.proxy_url or os.getenv("PROXY_URL", "http://127.0.0.1:7897")
    config["input_csv"] = args.input_csv or os.getenv("INPUT_CSV_PATH", AIR_NRM_QUERY_CSV)
    config["output_csv"] = args.output_csv or os.getenv("OUTPUT_CSV_PATH", "results.csv")
    config["chunk_size"] = args.chunk_size or int(os.getenv("CHUNK_SIZE", "11"))
    config["skip_rows"] = args.skip_rows if args.skip_rows is not None else int(os.getenv("SKIP_ROWS", "0"))
    config["timeout"] = args.timeout or int(os.getenv("CODE_EXECUTION_TIMEOUT", "300"))
    config["max_retries"] = args.max_retries or int(os.getenv("MAX_UPLOAD_RETRIES", "3"))
    default_max_tokens = "65536" if config["model_type"] == "gemini" else "16384"
    config["max_tokens"] = (
        args.max_tokens
        if args.max_tokens is not None
        else int(os.getenv("MAX_OUTPUT_TOKENS", default_max_tokens))
    )
    config["data_base_dir"] = args.data_base_dir or os.getenv("DATA_BASE_DIR", "")

    # Proxy: direct connection by default; --if_proxy explicitly enables it.
    if args.no_proxy:
        config["if_proxy"] = False
    elif args.if_proxy:
        config["if_proxy"] = True
    else:
        config["if_proxy"] = False

    # Model name: CLI > .env > provider default
    default_models = {
        "gemini": "gemini-3.1-pro-preview",
        "openai": "gpt-5.4",
        "claude": "claude-opus-4-6",
    }
    env_model = os.getenv("MODEL_NAME", "")
    config["model_name"] = args.model_name or (env_model if env_model else default_models.get(config["model_type"], ""))

    # API keys
    config["gemini_api_key"] = (
        os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    )
    config["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
    config["claude_api_key"] = os.getenv("CLAUDE_API_KEY", "")

    return config


# ==========================================
# 2. Proxy Setup
# ==========================================
def setup_proxy(config):
    if config["if_proxy"]:
        proxy_url = config["proxy_url"]
        for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
            os.environ[var] = proxy_url
        print(f"Proxy enabled: {proxy_url}")
    else:
        for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
            os.environ.pop(var, None)
        print("Proxy disabled.")


# ==========================================
# 3. LLM Client Abstraction
# ==========================================
class LLMClient:
    """Base class for LLM providers."""

    def __init__(self, config):
        self.config = config
        self.max_retries = config["max_retries"]

    def generate_with_metadata(self, prompt, file_paths, system_instructions=None):
        raise NotImplementedError

    def generate(self, prompt, file_paths):
        """Backward-compatible text-only interface used by the CSV runner."""
        return self.generate_with_metadata(prompt, file_paths)["text"]

    def upload_files(self, file_paths):
        """Default: no file upload needed. Override in subclasses that support it."""
        return []

    def cleanup_files(self, uploaded_files):
        """Default: no cleanup needed."""
        pass


def response_part_to_json(value):
    """Convert SDK response objects into JSON-safe diagnostic data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): response_part_to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [response_part_to_json(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", exclude_none=True)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            k: response_part_to_json(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return str(value)


def prompt_with_file_contents(prompt, file_paths):
    """Put every CSV byte into the one model request, with explicit filenames."""
    sections = [
        "The complete contents of the benchmark CSV files follow. "
        "Use only these filenames and do not invent merged or derived files."
    ]
    for path in file_paths:
        with open(path, "r", encoding="utf-8-sig") as file_obj:
            contents = file_obj.read()
        sections.append(
            f"\n### DATA FILE: {os.path.basename(path)}\n"
            f"```csv\n{contents}\n```"
        )
    sections.append("\n### TASK\n" + prompt)
    return "\n".join(sections)


class GeminiClient(LLMClient):
    def __init__(self, config):
        super().__init__(config)
        from google import genai
        from google.genai import types

        self.genai = genai
        self.types = types
        # One workflow stage means one HTTP attempt: disable the SDK's default
        # retry policy so a transient API failure cannot silently resend it.
        self.client = genai.Client(
            api_key=config["gemini_api_key"],
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        self.model_name = config["model_name"]

    def generate_with_metadata(self, prompt, file_paths, system_instructions=None):
        started = time.perf_counter()
        generation_config = {
            "system_instruction": (
                system_instructions
                if system_instructions is not None
                else GEMINI_SYSTEM_INSTRUCTIONS
            ),
            "max_output_tokens": self.config.get("max_tokens", 16384),
        }
        # Gemini 3.x is tuned for its default sampling settings. Keep these
        # unset unless a future profile deliberately opts into overrides.
        if self.config.get("temperature") is not None:
            generation_config["temperature"] = self.config["temperature"]
        if self.config.get("top_p") is not None:
            generation_config["top_p"] = self.config["top_p"]
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt_with_file_contents(prompt, file_paths),
            config=self.types.GenerateContentConfig(**generation_config),
        )
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = (
            (getattr(usage, "prompt_token_count", 0) or 0)
            + (getattr(usage, "tool_use_prompt_token_count", 0) or 0)
        )
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        reasoning_tokens = getattr(usage, "thoughts_token_count", 0) or 0
        cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or 0
        text_parts = []
        candidate_diagnostics = []
        finish_reasons = []
        for candidate in getattr(response, "candidates", []) or []:
            candidate_diagnostics.append(response_part_to_json(candidate))
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason is not None:
                finish_reasons.append(str(finish_reason).split(".")[-1])
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text and not getattr(part, "thought", False):
                    text_parts.append(text)
        normal_reasons = {"STOP", "FINISH_REASON_UNSPECIFIED"}
        provider_status = (
            "completed"
            if not finish_reasons or set(finish_reasons) <= normal_reasons
            else "incomplete"
        )
        return {
            "text": "\n".join(text_parts),
            "model": getattr(response, "model_version", None) or self.model_name,
            "provider_response_id": getattr(response, "response_id", None),
            "provider_status": provider_status,
            "finish_reasons": finish_reasons,
            "provider_response_parts": candidate_diagnostics,
            "prompt_feedback": response_part_to_json(
                getattr(response, "prompt_feedback", None)
            ),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_prompt_tokens": cached_tokens,
            "total_tokens": total_tokens or (
                prompt_tokens + completion_tokens + reasoning_tokens
            ),
            "token_source": "provider",
            "tool_calls": 0,
            "latency_s": round(time.perf_counter() - started, 3),
        }

class OpenAIClient(LLMClient):
    """Uses one tool-free Responses API request containing all five CSVs."""

    def __init__(self, config):
        super().__init__(config)
        from openai import OpenAI
        # OpenAI's SDK retries twice by default; this benchmark deliberately
        # records the first failure instead of issuing another model request.
        self.client = OpenAI(api_key=config["openai_api_key"], max_retries=0)
        self.model_name = config["model_name"]

    def _extract_response_text(self, response):
        if getattr(response, "output_text", None):
            return response.output_text

        response_parts = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for block in getattr(item, "content", []) or []:
                if getattr(block, "type", None) == "output_text":
                    response_parts.append(block.text)
        return "\n".join(response_parts)

    def generate_with_metadata(self, prompt, file_paths, system_instructions=None):
        started = time.perf_counter()
        content = [{
            "type": "input_text",
            "text": prompt_with_file_contents(prompt, file_paths),
        }]
        request = {
            "model": self.model_name,
            "instructions": (
                system_instructions
                if system_instructions is not None
                else SYSTEM_INSTRUCTIONS
            ),
            "input": [{"role": "user", "content": content}],
        }
        # Null sampling settings mean use the provider default. GPT-5 with
        # reasoning enabled rejects explicit temperature/top_p parameters.
        for parameter, value in (
            ("temperature", self.config.get("temperature", 0.0)),
            ("top_p", self.config.get("top_p", 1.0)),
            ("max_output_tokens", self.config.get("max_tokens", 16384)),
        ):
            if value is not None:
                request[parameter] = value
        if self.config.get("reasoning_effort"):
            request["reasoning"] = {"effort": self.config["reasoning_effort"]}
        response = self.client.responses.create(**request)
        usage = getattr(response, "usage", None)
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0
        cached_tokens = getattr(input_details, "cached_tokens", 0) or 0
        reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        status = getattr(response, "status", None) or "unknown"
        error = getattr(response, "error", None)
        return {
            "text": self._extract_response_text(response),
            "model": getattr(response, "model", None) or self.model_name,
            "provider_response_id": getattr(response, "id", None),
            "provider_status": status,
            "finish_reasons": [status],
            "provider_error": response_part_to_json(error),
            "provider_response_parts": response_part_to_json(
                getattr(response, "output", []) or []
            ),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_prompt_tokens": cached_tokens,
            "total_tokens": total_tokens or prompt_tokens + completion_tokens,
            "token_source": "provider",
            "tool_calls": 0,
            "latency_s": round(time.perf_counter() - started, 3),
        }


class ClaudeClient(LLMClient):
    """Uses Claude Messages API with Files API upload and code_execution tool."""

    def __init__(self, config):
        super().__init__(config)
        import anthropic
        self.client = anthropic.Anthropic(api_key=config["claude_api_key"])
        self.model_name = config["model_name"]

    def generate(self, prompt, file_paths):
        # Upload files via Files API and build content blocks
        uploaded_file_ids = []
        content_blocks = [{"type": "text", "text": prompt}]

        for path in file_paths:
            if os.path.exists(path):
                for attempt in range(self.max_retries):
                    try:
                        with open(path, "rb") as f:
                            file_obj = self.client.beta.files.upload(file=f)
                        uploaded_file_ids.append(file_obj.id)
                        content_blocks.append({
                            "type": "container_upload",
                            "file_id": file_obj.id,
                        })
                        print(f"  Uploaded {os.path.basename(path)} -> {file_obj.id}")
                        break
                    except Exception as e:
                        print(f"  Upload {path} failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                        if attempt < self.max_retries - 1:
                            time.sleep(5)
                        else:
                            print(f"  Skipping file {path} after max retries.")
            else:
                print(f"  Warning: file not found {path}")

        try:
            for attempt in range(self.max_retries):
                try:
                    response = self.client.beta.messages.create(
                        model=self.model_name,
                        betas=["files-api-2025-04-14"],
                        max_tokens=16384,
                        system=(
                            "You are an expert in operations research and mathematical optimization. "
                            "When asked to solve OR problems, provide a mathematical model and Python code using gurobipy. "
                            "IMPORTANT: Always include the complete gurobipy Python code in your final text response "
                            "inside a ```python ... ``` code block so it can be extracted and run locally."
                        ),
                        messages=[{"role": "user", "content": content_blocks}],
                        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
                    )
                    # Extract text and code execution content from response
                    text_parts = []
                    code_execution_parts = []
                    for block in response.content:
                        if block.type == "text":
                            text_parts.append(block.text)
                        elif block.type == "code_execution_tool_use":
                            # Capture code that Claude wrote and executed
                            if hasattr(block, "input") and isinstance(block.input, dict):
                                code = block.input.get("code", "")
                                if code:
                                    code_execution_parts.append(code)
                        elif block.type == "bash_code_execution_tool_use":
                            # Capture bash commands Claude executed
                            if hasattr(block, "input") and isinstance(block.input, dict):
                                cmd = block.input.get("command", "")
                                if cmd:
                                    code_execution_parts.append(cmd)
                        elif block.type == "bash_code_execution_tool_result":
                            # Download any generated files from sandbox
                            if hasattr(block, "content"):
                                result_content = block.content
                                if hasattr(result_content, "content"):
                                    for item in result_content.content:
                                        if hasattr(item, "file_id") and item.file_id:
                                            try:
                                                meta = self.client.beta.files.retrieve_metadata(item.file_id)
                                                if meta.filename and meta.filename.endswith(".py"):
                                                    file_data = self.client.beta.files.download(item.file_id)
                                                    py_code = file_data.read().decode("utf-8")
                                                    code_execution_parts.append(py_code)
                                                    print(f"  Downloaded sandbox file: {meta.filename}")
                                            except Exception as e:
                                                print(f"  Warning: could not download sandbox file: {e}")

                    # Build full response: text blocks first
                    full_response = "\n".join(text_parts)

                    # If no python code block found in text, append code from execution
                    if "```python" not in full_response and code_execution_parts:
                        # Find the longest code block as the main solution
                        best_code = max(code_execution_parts, key=len)
                        full_response += "\n\n```python\n" + best_code + "\n```"

                    return full_response
                except Exception as e:
                    print(f"Claude API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(5)
                    else:
                        raise
        finally:
            # Cleanup uploaded files
            for fid in uploaded_file_ids:
                try:
                    self.client.beta.files.delete(fid)
                except Exception:
                    pass


def create_client(config):
    """Factory function to create the appropriate LLM client."""
    clients = {
        "gemini": GeminiClient,
        "openai": OpenAIClient,
        "claude": ClaudeClient,
    }
    client_cls = clients.get(config["model_type"])
    if client_cls is None:
        raise ValueError(f"Unknown model_type: {config['model_type']}")
    return client_cls(config)


# ==========================================
# 4. Utility Functions
# ==========================================
def extract_obj_from_gurobi_log(log_text):
    """Extract optimal objective from Gurobi console output."""
    pattern = r"(?:Optimal objective|Best objective|Objective value[:]?)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    match = re.search(pattern, log_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def validate_gurobi_code(python_code):
    """Return an extraction error string, or an empty string for valid code."""
    if not python_code or not python_code.strip():
        return "no_code_block"

    try:
        tree = ast.parse(python_code)
    except SyntaxError as exc:
        line = exc.lineno or "?"
        return f"python_syntax_error: line {line}: {exc.msg}"

    has_gurobi_import = any(
        (
            isinstance(node, ast.Import)
            and any(alias.name == "gurobipy" for alias in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module == "gurobipy"
        )
        for node in ast.walk(tree)
    )
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    has_model = any(
        (isinstance(func, ast.Name) and func.id == "Model")
        or (isinstance(func, ast.Attribute) and func.attr == "Model")
        for func in calls
    )
    has_optimize = any(
        isinstance(func, ast.Attribute) and func.attr == "optimize"
        for func in calls
    )

    missing = []
    if not has_gurobi_import:
        missing.append("gurobipy import")
    if not has_model:
        missing.append("Model()")
    if not has_optimize:
        missing.append("optimize()")
    if missing:
        return "no_gurobi_code: missing " + ", ".join(missing)
    return ""


def extract_code_from_response(response_text):
    """Extract and validate the best final gurobipy block from a response."""
    if not response_text or not str(response_text).strip():
        return "", "", "empty_response"

    response_text = str(response_text)
    python_heading = re.search(
        r"(?im)^#{2,6}\s*Python\s+Code\s*:?\s*$",
        response_text,
    )
    math_heading = re.search(
        r"(?im)^#{2,6}\s*Mathematical\s+Model\s*:?\s*$",
        response_text,
    )

    math_model = ""
    if math_heading:
        math_end = python_heading.start() if python_heading else len(response_text)
        math_model = response_text[math_heading.end():math_end].strip()

    code_section = (
        response_text[python_heading.end():]
        if python_heading
        else response_text
    )

    fenced_pattern = re.compile(
        r"```[ \t]*(?P<label>[^\r\n`]*)[ \t]*\r?\n"
        r"(?P<code>.*?)```",
        re.DOTALL,
    )
    candidates = []
    for match in fenced_pattern.finditer(code_section):
        label = match.group("label").strip().lower()
        if label in ("python", "py", ""):
            candidates.append((label, match.group("code").strip()))

    # Accept a final unclosed Python fence when the code itself is complete.
    if not candidates:
        unclosed = re.search(
            r"```[ \t]*(?:python|py)[ \t]*\r?\n(?P<code>.+)\Z",
            code_section,
            re.IGNORECASE | re.DOTALL,
        )
        if unclosed:
            candidates.append(("python", unclosed.group("code").strip()))

    # Last-resort recovery for complete, unfenced code returned after the
    # requested heading or beginning at a gurobipy import.
    if not candidates:
        if python_heading:
            raw_code = code_section.strip()
        else:
            code_start = re.search(
                r"(?m)^(?:from\s+gurobipy\b|import\s+gurobipy\b)",
                code_section,
            )
            raw_code = code_section[code_start.start():].strip() if code_start else ""
        if raw_code:
            candidates.append(("unfenced", raw_code))

    if not candidates:
        return math_model, "", "no_code_block"

    def candidate_score(candidate):
        label, code = candidate
        lowered = code.lower()
        return (
            1 if label in ("python", "py") else 0,
            sum(marker in lowered for marker in ("gurobipy", "model(", "optimize(")),
            len(code),
        )

    candidates.sort(key=candidate_score, reverse=True)
    first_error = "no_code_block"
    for _, candidate_code in candidates:
        error = validate_gurobi_code(candidate_code)
        if not error:
            return math_model, candidate_code, ""
        if first_error == "no_code_block":
            first_error = error

    return math_model, candidates[0][1], first_error


def replace_file_paths_in_code(python_code, dataset_addresses):
    """Replace placeholder CSV paths in generated code with real dataset paths."""
    if not python_code or not dataset_addresses:
        return python_code

    csvs_in_code = list(set(re.findall(r"['\"]([^'\"]+\.csv)['\"]", python_code)))
    if len(dataset_addresses) == 1:
        real_path = dataset_addresses[0]
        for c in csvs_in_code:
            python_code = python_code.replace(f'"{c}"', f'"{real_path}"')
            python_code = python_code.replace(f"'{c}'", f"'{real_path}'")
    else:
        for c in csvs_in_code:
            best_match_path = None
            c_basename = os.path.basename(c).lower()

            # A quoted diagnostic such as
            # "OD {od} not found in v1.csv or v2.csv" is prose, not a file
            # argument. Treat only compact filename/path literals as mapping
            # candidates; otherwise harmless error messages produce alarming
            # "ambiguous match" warnings during every benchmark run.
            looks_like_path = (
                not any(ch.isspace() for ch in c)
                and "{" not in c
                and "}" not in c
            )
            if not looks_like_path:
                continue

            # Exact basename matching is required for Air-NRM, which contains
            # both flight.csv and flight_capacity.csv.
            exact_matches = [
                real_path for real_path in dataset_addresses
                if os.path.basename(real_path).lower() == c_basename
            ]
            if len(exact_matches) == 1:
                best_match_path = exact_matches[0]
            elif len(exact_matches) > 1:
                print(f"Warning: {c} has multiple exact matches in Dataset_address!")
            else:
                c_name_no_ext = os.path.splitext(c_basename)[0]
                fallback_matches = []
                for real_path in dataset_addresses:
                    real_name_no_ext = os.path.splitext(
                        os.path.basename(real_path).lower()
                    )[0]
                    if real_name_no_ext in c_name_no_ext or c_name_no_ext in real_name_no_ext:
                        fallback_matches.append(real_path)
                if len(fallback_matches) == 1:
                    best_match_path = fallback_matches[0]
                elif len(fallback_matches) > 1:
                    print(f"Warning: {c} has ambiguous matches in Dataset_address!")
            if best_match_path:
                python_code = python_code.replace(f'"{c}"', f'"{best_match_path}"')
                python_code = python_code.replace(f"'{c}'", f"'{best_match_path}'")
            else:
                print(f"Warning: {c} in code has no match in Dataset_address!")

    return python_code


def run_generated_code(python_code, timeout):
    """Execute the generated Python code locally and return (stdout, stderr, returncode)."""
    try:
        # A unique working directory prevents concurrent model runs from
        # overwriting each other's Python files or generated solver artifacts.
        with tempfile.TemporaryDirectory(prefix="airnrm_agentic_") as temp_dir:
            temp_py_file = os.path.join(temp_dir, "generated_solution.py")
            with open(temp_py_file, "w", encoding="utf-8") as f:
                f.write(python_code)

            process = subprocess.run(
                [sys.executable, temp_py_file],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired:
        return "", f"Execution timed out after {timeout}s", -1


# ==========================================
# 5. Main Processing Logic
# ==========================================
def build_prompt(query):
    return f"""Below is an operations research question. Build a mathematical model and corresponding python code using 'gurobipy' that appropriately addresses the question.

# Question

{query}

# Required Response Format

{RESPONSE_FORMAT_REQUIREMENTS}
"""


def get_dataset_addresses(row, config):
    """Resolve per-row data files, falling back to Air-NRM's shared CSVs."""
    if "Dataset_address" in row.index and pd.notna(row["Dataset_address"]):
        raw_addresses = str(row["Dataset_address"]).strip()
    else:
        raw_addresses = ""

    if raw_addresses:
        dataset_addresses = raw_addresses.split()
        base_dir = config.get("data_base_dir", "")
        if base_dir:
            base_dir = os.path.abspath(base_dir)
            dataset_addresses = [
                os.path.abspath(os.path.join(base_dir, path))
                if not os.path.isabs(path)
                else os.path.abspath(path)
                for path in dataset_addresses
            ]
    else:
        dataset_addresses = list(AIR_NRM_DATA_FILES)

    missing_files = [path for path in dataset_addresses if not os.path.isfile(path)]
    if missing_files:
        raise FileNotFoundError(
            "Dataset file(s) not found: " + ", ".join(missing_files)
        )
    return dataset_addresses


def process_row(client, row, config):
    """Process a single row: call LLM, extract code, run it, compare result."""
    result = {
        "LLM_Response": "",
        "Math_Model": "",
        "Python_Code": "",
        "Generated_Obj": float("nan"),
        "Error_Log": "",
        "Count": 0,
    }

    try:
        query = row["Query"]
        raw_obj_str = re.sub(r"[^0-9.\-+eE]", "", str(row["Label-objective"]))
        label_obj = float(raw_obj_str)
        dataset_addresses = get_dataset_addresses(row, config)

        prompt = build_prompt(query)

        print("Requesting LLM response...")
        response_text = client.generate(prompt, dataset_addresses)
        result["LLM_Response"] = response_text or ""

        math_model, python_code, extraction_error = extract_code_from_response(
            response_text
        )
        result["Math_Model"] = math_model

        if not extraction_error:
            python_code = replace_file_paths_in_code(
                python_code,
                dataset_addresses,
            )
        result["Python_Code"] = python_code

        if extraction_error:
            result["Error_Log"] = extraction_error
            print(f"  Extraction error: {extraction_error}")
        elif python_code:
            print("Executing generated Gurobi code...")
            stdout, stderr, returncode = run_generated_code(python_code, config["timeout"])

            if returncode == 0:
                gen_obj = extract_obj_from_gurobi_log(stdout)
                if gen_obj is not None:
                    result["Generated_Obj"] = gen_obj
                    if math.isclose(gen_obj, label_obj, rel_tol=1e-4):
                        result["Count"] = 1
                        print(f"  CORRECT (generated: {gen_obj:.4f}, label: {label_obj:.4f})")
                    else:
                        result["Count"] = 0
                        print(f"  WRONG   (generated: {gen_obj:.4f}, label: {label_obj:.4f})")
                else:
                    result["Error_Log"] = (
                        "objective_parse_error: could not extract optimal "
                        "objective from output"
                    )
            else:
                execution_details = (stderr or stdout or "no subprocess output").strip()
                result["Error_Log"] = f"execution_error: {execution_details}"
                print(f"  Code error: {stderr[:200]}...")
        else:
            result["Error_Log"] = "no_code_block"

    except Exception as e:
        result["Error_Log"] = f"request_or_processing_error: {e}"
        result["Generated_Obj"] = float("nan")
        result["Count"] = 0
        print(f"  Encounter error (skipping): {e}")

    return result


def main():
    args = parse_args()
    config = get_config(args)

    # Print configuration
    print("=" * 60)
    print("LLM4OR Benchmark - Universal Runner")
    print("=" * 60)
    print(f"  Model type : {config['model_type']}")
    print(f"  Model name : {config['model_name']}")
    print(f"  Proxy      : {'ON (' + config['proxy_url'] + ')' if config['if_proxy'] else 'OFF'}")
    print(f"  Input CSV  : {config['input_csv']}")
    print(f"  Output CSV : {config['output_csv']}")
    print(f"  Chunk size : {config['chunk_size']}")
    print(f"  Skip rows  : {config['skip_rows']}")
    print(f"  Timeout    : {config['timeout']}s")
    print(f"  Max tokens : {config['max_tokens']}")
    print("=" * 60)

    # Ensure results directory exists
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Setup proxy
    setup_proxy(config)

    # Create LLM client
    client = create_client(config)

    # Load dataset
    print(f"\nLoading dataset: {config['input_csv']}")
    df_full = pd.read_csv(config["input_csv"])
    print(f"Total rows: {len(df_full)}")

    required_columns = {"Query", "Label-objective"}
    missing_columns = required_columns.difference(df_full.columns)
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    skip = config["skip_rows"]
    if skip > 0:
        df_full = df_full.iloc[skip:].reset_index(drop=True)
        print(f"Skipped first {skip} rows, remaining: {len(df_full)}")

    total_rows = len(df_full)
    chunk_size = config["chunk_size"]
    total_chunks = math.ceil(total_rows / chunk_size)
    chunk_files = []

    # Process in chunks
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)

        print(f"\n{'=' * 50}")
        print(f"Batch {chunk_idx + 1}/{total_chunks} (rows {start_idx} to {end_idx - 1})")
        print(f"{'=' * 50}")

        chunk_df = df_full.iloc[start_idx:end_idx].copy()
        for col in ["LLM_Response", "Math_Model", "Python_Code", "Generated_Obj", "Error_Log"]:
            chunk_df[col] = ""
        chunk_df["Count"] = 0

        for index, row in chunk_df.iterrows():
            current_pos = index - start_idx + 1
            print(f"\n[Global: {index + 1}/{total_rows} | Batch: {current_pos}/{len(chunk_df)}] Processing...")

            result = process_row(client, row, config)
            for key, val in result.items():
                chunk_df.at[index, key] = val

        # Save chunk
        chunk_path = os.path.join(results_dir, f"temp_chunk_{config['model_type']}_{chunk_idx + 1}.csv")
        chunk_df.to_csv(chunk_path, index=False, encoding="utf-8")
        chunk_files.append(chunk_path)
        print(f"Batch {chunk_idx + 1} saved to: {chunk_path}")

    # Merge all chunks
    print(f"\n{'=' * 50}")
    print("Merging all batches...")
    all_dfs = [pd.read_csv(f) for f in chunk_files]
    final_df = pd.concat(all_dfs, ignore_index=True)
    output_path = os.path.join(results_dir, config["output_csv"])
    final_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Final results saved to: {output_path}")

    # Cleanup temp files
    for f in chunk_files:
        if os.path.exists(f):
            os.remove(f)
    print("Temp files cleaned up.")

    # Print summary
    correct = final_df["Count"].sum()
    total = len(final_df)
    print(f"\nAccuracy: {correct}/{total} = {correct / total * 100:.2f}%")


if __name__ == "__main__":
    main()
