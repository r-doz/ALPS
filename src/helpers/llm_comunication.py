import os
import requests
import json
import re
import time

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

PROGRAMS_SCHEMA = {
    "type": "object",
    "required": ["programs"],
    "properties": {
        "programs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "hypothesis", "structure", "program"],
                "properties": {
                    "id": {"type": "integer"},
                    "hypothesis": {"type": "string"},
                    "structure": {"type": "string"},
                    "program": {"type": "string"}
                }
            }
        }
    }
}

def _extract_response_text(resp_json: dict) -> str:
    """Get model text from common Ollama response shapes."""
    if not isinstance(resp_json, dict):
        return ""
    if isinstance(resp_json.get("response"), str):
        return resp_json["response"]
    message = resp_json.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    # Some models can emit reasoning text under "thinking" when "content" is empty.
    if isinstance(message, dict) and isinstance(message.get("thinking"), str):
        return message["thinking"]
    if isinstance(resp_json.get("content"), str):
        return resp_json["content"]
    return ""


def _salvage_unstructured_output(
    raw_text: str,
    model: str,
    system_prompt: str,
    use_chat: bool,
    request_timeout: int,
    max_http_retries: int,
    num_predict: int,
):
    """Ask the model to convert unstructured text into the expected JSON schema."""
    salvage_user_message = (
        "Convert the text below into ONLY valid JSON using this exact top-level shape:\n"
        "{\"programs\": [{\"id\": int, \"hypothesis\": str, \"structure\": str, \"program\": str}]}\n\n"
        "Rules:\n"
        "- Keep only recoverable program entries.\n"
        "- If an id is missing, assign sequential ids starting from 1.\n"
        "- Output JSON only (no prose, no markdown).\n\n"
        f"Text to convert:\n{raw_text}"
    )

    if use_chat:
        salvage_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": salvage_user_message},
            ],
            "stream": False,
            "format": PROGRAMS_SCHEMA,
            "options": {"temperature": 0.0, "num_predict": num_predict},
        }
    else:
        salvage_prompt = f"""{system_prompt}

User: {salvage_user_message}
Assistant:"""
        salvage_payload = {
            "model": model,
            "prompt": salvage_prompt,
            "stream": False,
            "format": PROGRAMS_SCHEMA,
            "options": {"temperature": 0.0, "num_predict": num_predict},
        }

    salvage_resp = _post_ollama(
        salvage_payload,
        use_chat=use_chat,
        request_timeout=request_timeout,
        max_http_retries=max_http_retries,
    )
    salvage_raw = _extract_response_text(salvage_resp)
    salvage_parsed = _extract_json(salvage_raw)
    return _normalize_programs_payload(salvage_parsed)

def _extract_json(raw: str):
    """Parse direct JSON first, then recover JSON object/array embedded in text."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty model response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise

def _normalize_programs_payload(obj):
    """Accept either {'programs': [...]} or [...] and normalize."""
    if isinstance(obj, list):
        return {"programs": obj}
    if isinstance(obj, dict) and "programs" in obj and isinstance(obj["programs"], list):
        return obj
    raise ValueError(f"Unexpected JSON shape: {type(obj)}")

def _post_ollama(payload: dict, use_chat: bool = True, request_timeout: int = 1000, max_http_retries: int = 3) -> dict:
    """POST to Ollama with retry on transient timeout/connection errors."""
    url = OLLAMA_CHAT_URL if use_chat else OLLAMA_GENERATE_URL
    last_exc = None
    for attempt in range(max_http_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=request_timeout)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt >= max_http_retries:
                break
            time.sleep(1.5 * (attempt + 1))
    raise last_exc

def call_ollama(
    user_message: str,
    system_prompt: str,
    model: str = OLLAMA_MODEL,
    temperature: float = 0.2,
    require_json: bool = True,
    max_retries: int = 1,
    use_chat: bool = True,
    request_timeout: int = 600,
    max_http_retries: int = 3,
):
    model = model.strip()
    # JSON tasks need a larger token budget; otherwise gpt-oss can stop before content is emitted.
    num_predict = 4000 if require_json else 1200

    if use_chat:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict
            },
        }
    else:
        prompt = f"""{system_prompt}

User: {user_message}
Assistant:"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict
            },
        }

    if require_json:
        payload["format"] = PROGRAMS_SCHEMA

    try:
        resp_json = _post_ollama(
            payload,
            use_chat=use_chat,
            request_timeout=request_timeout,
            max_http_retries=max_http_retries,
        )
    except Exception as exc:
        print(f"\n❌ Ollama transport error: {exc}")
        return {
            "programs": [],
            "parse_error": f"Transport error: {exc}",
            "raw_response": "",
            "response_json": {},
        }

    raw = _extract_response_text(resp_json)

    if not require_json:
        return raw

    for attempt in range(max_retries + 1):
        try:
            parsed = _extract_json(raw)
            return _normalize_programs_payload(parsed)
        except Exception as exc:
            print(f"\n❌ Ollama response parse error: {exc}")
            if attempt >= max_retries:
                # Last chance: convert any unstructured text into valid JSON.
                if raw.strip():
                    try:
                        return _salvage_unstructured_output(
                            raw_text=raw,
                            model=model,
                            system_prompt=system_prompt,
                            use_chat=use_chat,
                            request_timeout=request_timeout,
                            max_http_retries=max_http_retries,
                            num_predict=num_predict,
                        )
                    except Exception as salvage_exc:
                        print(f"\n❌ Salvage attempt failed: {salvage_exc}")
                        return {
                            "programs": [],
                            "parse_error": f"{exc}; salvage failed: {salvage_exc}",
                            "raw_response": raw,
                            "response_json": resp_json,
                        }
                return {
                    "programs": [],
                    "parse_error": str(exc),
                    "raw_response": raw,
                    "response_json": resp_json,
                }

            retry_user_message = (
                "Return ONLY valid JSON with this exact top-level shape:\n"
                "{\"programs\": [{\"id\": int, \"hypothesis\": str, \"structure\": str, \"program\": str}]}\n\n"
                f"Task:\n{user_message}"
            )

            if use_chat:
                retry_payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": retry_user_message},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": num_predict},
                }
            else:
                retry_prompt = f"""{system_prompt}

User: {retry_user_message}
Assistant:"""
                retry_payload = {
                    "model": model,
                    "prompt": retry_prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": num_predict},
                }

            try:
                resp_json = _post_ollama(
                    retry_payload,
                    use_chat=use_chat,
                    request_timeout=request_timeout,
                    max_http_retries=max_http_retries,
                )
                raw = _extract_response_text(resp_json)
            except Exception as retry_exc:
                print(f"\n❌ Ollama transport error on retry: {retry_exc}")
                return {
                    "programs": [],
                    "parse_error": f"Transport error on retry: {retry_exc}",
                    "raw_response": raw,
                    "response_json": resp_json,
                }