"""Smoke test: OpenAI tool calling against the local server.

Sends a get_weather tool request (non-stream + stream) to each loaded model
and reports whether a well-formed tool_calls response comes back.

Run: .\.venv\Scripts\python.exe scripts\smoke_tools.py
"""

import json
import urllib.request

BASE = "http://localhost:8000/v1"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]


def _post(path: str, payload: dict):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=300)


def models():
    with urllib.request.urlopen(f"{BASE}/models") as r:
        return [m["id"] for m in json.load(r)["data"]]


def test_nonstream(model: str) -> bool:
    body = {
        "model": model,
        "messages": [{"role": "user",
                      "content": "What is the weather in Amsterdam in celsius?"}],
        "tools": TOOLS,
        "max_tokens": 256,
    }
    with _post("/chat/completions", body) as r:
        data = json.load(r)
    msg = data["choices"][0]["message"]
    finish = data["choices"][0]["finish_reason"]
    calls = msg.get("tool_calls") or []
    ok = (finish == "tool_calls" and len(calls) == 1
          and calls[0]["function"]["name"] == "get_weather"
          and "msterdam" in calls[0]["function"]["arguments"])
    print(f"  non-stream: finish={finish} calls={json.dumps(calls)[:120]} "
          f"content={str(msg.get('content'))[:60]!r} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_stream(model: str) -> bool:
    body = {
        "model": model,
        "messages": [{"role": "user",
                      "content": "What is the weather in Amsterdam in celsius?"}],
        "tools": TOOLS,
        "max_tokens": 256,
        "stream": True,
    }
    calls, finish = [], None
    with _post("/chat/completions", body) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            choice = chunk["choices"][0]
            if choice["delta"].get("tool_calls"):
                calls += choice["delta"]["tool_calls"]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    ok = (finish == "tool_calls" and len(calls) == 1
          and calls[0]["function"]["name"] == "get_weather")
    print(f"  stream:     finish={finish} calls={json.dumps(calls)[:120]} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def test_tool_result_roundtrip(model: str) -> bool:
    """Second turn: feed the tool result back, expect a text answer."""
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is the weather in Amsterdam?"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_abc123", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": "{\"city\": \"Amsterdam\"}"}}]},
            {"role": "tool", "tool_call_id": "call_abc123",
             "content": "{\"temp_c\": 18, \"condition\": \"sunny\"}"},
        ],
        "tools": TOOLS,
        "max_tokens": 256,
    }
    with _post("/chat/completions", body) as r:
        data = json.load(r)
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    ok = "18" in content and not msg.get("tool_calls")
    print(f"  roundtrip:  content={content[:80]!r} -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = {}
    for m in models():
        print(f"\n=== {m} ===")
        results[m] = all([test_nonstream(m), test_stream(m),
                          test_tool_result_roundtrip(m)])
    print("\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}  {m}"
                           for m, ok in results.items()))
