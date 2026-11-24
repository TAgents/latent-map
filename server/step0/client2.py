import json, urllib.request
U = "http://127.0.0.1:8766/v1/chat/completions"
def post(body):
    req = urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=300)
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the project and return its text.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
q = {"role": "user", "content": "What version is declared in pyproject.toml? Use the tools."}
m = json.loads(post({"messages": [q], "tools": TOOLS, "max_tokens": 300}).read())["choices"][0]
print("tool turn:", json.dumps(m)[:700])
m = json.loads(post({"messages": [{"role": "user", "content": "Why is the sky blue? One sentence."}], "max_tokens": 300}).read())["choices"][0]
print("chat turn:", json.dumps(m)[:900])
c = r = ""; tc = []
for l in post({"messages": [q], "tools": TOOLS, "max_tokens": 300, "stream": True}):
    l = l.decode()
    if l.startswith("data: {"):
        d = json.loads(l[6:])["choices"][0]
        c += d["delta"].get("content") or ""; r += d["delta"].get("reasoning") or ""; tc += d["delta"].get("tool_calls") or []
        fin = d.get("finish_reason") or None
print("stream:", repr(c), "| reasoning:", repr(r[:120]), "| tool_calls:", tc, "| finish:", fin)
