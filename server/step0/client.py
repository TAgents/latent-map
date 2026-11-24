import json, time, threading, urllib.request
U = "http://127.0.0.1:8766/v1/chat/completions"
def post(body, label):
    t = time.time()
    req = urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=500).read())
        m = r["choices"][0]["message"]
        print(label, round(time.time() - t, 1), "s", r.get("usage"), repr((m.get("content") or "")[-260:]), m.get("tool_calls"), flush=True)
    except urllib.error.HTTPError as e:
        print(label, "HTTP", e.code, e.read()[:400], flush=True)
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the project and return its text.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
q = {"role": "user", "content": "What version is declared in pyproject.toml? Use the tools."}
res = {"role": "tool", "tool_call_id": "call_1", "content": '[project]\nname = "demo"\nversion = "0.4.2"\n'}
call = lambda args: {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": args}}]}
post({"messages": [q, call('{"path": "pyproject.toml"}'), res], "tools": TOOLS, "max_tokens": 200}, "tool-result, string args:")
post({"messages": [q, call({"path": "pyproject.toml"}), res], "tools": TOOLS, "max_tokens": 200}, "tool-result, object args:")
# long prompt: prefill speed, then the same prefix again with a different question
doc = open("../../spike/.venv/lib/python3.12/site-packages/mlx_lm/server.py").read()[:24000]
sysm = {"role": "system", "content": "You are a coding assistant. Here is a file:\n" + doc}
post({"messages": [sysm, {"role": "user", "content": "Name one class in the file. One word."}], "max_tokens": 120, "chat_template_kwargs": {"reasoning_effort": "low"}}, "long 1:")
post({"messages": [sysm, {"role": "user", "content": "Name one function in the file. One word."}], "max_tokens": 120, "chat_template_kwargs": {"reasoning_effort": "low"}}, "long 2 (same prefix):")
# a third turn that extends long 1 with the model's answer and a new question
post({"messages": [sysm, {"role": "user", "content": "Name one class in the file. One word."}, {"role": "assistant", "content": "ModelProvider"}, {"role": "user", "content": "And one more?"}], "max_tokens": 120, "chat_template_kwargs": {"reasoning_effort": "low"}}, "long 3 (extends long 1):")
# two at once
ts = [threading.Thread(target=post, args=({"messages": [{"role": "user", "content": c}], "max_tokens": 60, "chat_template_kwargs": {"reasoning_effort": "none"}}, f"concurrent {i}:")) for i, c in enumerate(["Say the word apple and nothing else.", "Say the word river and nothing else."])]
[t.start() for t in ts]; [t.join() for t in ts]
