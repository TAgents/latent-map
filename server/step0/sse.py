import sys, json
c=""; tc=[]; fr=None
for l in open(sys.argv[1]):
    if l.startswith("data: {"):
        d=json.loads(l[6:])["choices"][0]
        c+=d["delta"].get("content") or ""
        if d["delta"].get("reasoning"): c+="[R:"+d["delta"]["reasoning"]+"]"
        if d["delta"].get("tool_calls"): tc.append(d["delta"]["tool_calls"])
        fr=d.get("finish_reason") or fr
print(repr(c)); print("tool_calls:",tc,"finish:",fr)
