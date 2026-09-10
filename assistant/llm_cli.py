"""LLM provider backed by the Claude Code CLI (`claude -p`), authenticated with CLAUDE_CODE_OAUTH_TOKEN.

The CLI has no native tool-call API for our tools, so the model answers with a JSON object
{"say": "...", "actions": [{"tool": "...", "args": {...}}]} enforced by --json-schema.
History is rendered as text; tool results are fed back as text on the next call.
"""
import json, logging, os, shutil, subprocess, tempfile, uuid

log = logging.getLogger("assistant.cli")
CLAUDE_BIN = os.getenv("CLAUDE_BIN") or shutil.which("claude") or "claude"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "sonnet")
CLAUDE_TIMEOUT = int(os.getenv("CLAUDE_TIMEOUT", "90"))

SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "say": {"type": "string", "description": "что произнести вслух"},
        "actions": {"type": "array", "items": {"type": "object",
                    "properties": {"tool": {"type": "string"}, "args": {"type": "object"}},
                    "required": ["tool", "args"]}},
    },
    "required": ["say", "actions"],
})


def render_history(messages):
    lines = []
    for m in messages:
        role = m["role"]
        if role == "user":
            c = m["content"]
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if p.get("type") == "text") + " [к сообщению приложен кадр с камеры: FRAME_PATH]"
            lines.append(f"Человек: {c}")
        elif role == "assistant":
            if m.get("content"):
                lines.append(f"Ассистент: {m['content']}")
            for tc in m.get("tool_calls") or []:
                lines.append(f"Ассистент вызвал {tc['function']['name']}({tc['function']['arguments']})")
        elif role == "tool":
            lines.append(f"Результат {m.get('name', '')}: {m['content']}")
    return "\n".join(lines)


def tools_text(tools):
    if not tools:
        return "Инструментов нет: actions всегда пустой список."
    out = ["Доступные инструменты (кладёшь в actions, args по схеме):"]
    for t in tools:
        props = t["parameters"].get("properties", {})
        sig = ", ".join(f"{k}: {v.get('type', 'string')}" for k, v in props.items())
        out.append(f"- {t['name']}({sig}): {t['description']}")
    return "\n".join(out)


def chat(system, messages, tools=None, max_tokens=300, temperature=0.6):
    frame_path = None
    last = messages[-1] if messages else None
    if last and last["role"] == "user" and isinstance(last["content"], list):
        for p in last["content"]:
            if p.get("type") == "image_url":
                b64 = p["image_url"]["url"].split(",", 1)[1]
                fd, frame_path = tempfile.mkstemp(suffix=".jpg")
                with os.fdopen(fd, "wb") as f:
                    import base64
                    f.write(base64.b64decode(b64))
    history = render_history(messages[-14:]).replace("FRAME_PATH", frame_path or "")
    sysp = (system + "\n\n" + tools_text(tools) +
            "\nОтветь строго JSON по схеме: say - что сказать вслух (коротко, для озвучки), actions - список вызовов инструментов (может быть пустым). "
            "Если инструмент вернул результат в истории, используй его в say и не вызывай снова.")
    prompt = ("Диалог:\n" + history + "\n\nСформируй следующую реплику ассистента.") if history else "Поздоровайся."
    cmd = [CLAUDE_BIN, "-p", prompt, "--output-format", "json", "--system-prompt", sysp, "--json-schema", SCHEMA,
           "--no-session-persistence", "--model", CLAUDE_MODEL, "--max-turns", "3", "--tools", "Read" if frame_path else ""]
    if frame_path:
        prompt_note = f"\nКадр с камеры лежит в файле {frame_path}, прочитай его инструментом Read, если вопрос о том, что видно."
        cmd[2] = prompt + prompt_note
    env = dict(os.environ)
    env.setdefault("HOME", "/root")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=CLAUDE_TIMEOUT, env=env, cwd=tempfile.gettempdir())
    if frame_path:
        try:
            os.remove(frame_path)
        except OSError:
            pass
    try:
        d = json.loads(r.stdout)
    except Exception:
        raise RuntimeError(f"claude cli bad output rc={r.returncode}: {r.stdout[:300]} {r.stderr[:300]}")
    out = d.get("structured_output")
    if not out:
        res = d.get("result") or ""
        try:
            out = json.loads(res[res.index("{"):res.rindex("}") + 1])
        except Exception:
            if d.get("is_error"):
                raise RuntimeError(f"claude cli error: {d.get('errors') or d.get('subtype')} {r.stderr[:300]}")
            out = {"say": res.strip(), "actions": []}
    say = (out.get("say") or "").strip()
    calls = [{"id": f"call_{uuid.uuid4().hex[:8]}", "name": a.get("tool", ""), "args": a.get("args") or {}}
             for a in out.get("actions") or [] if isinstance(a, dict) and a.get("tool")]
    assistant_msg = {"role": "assistant", "content": say}
    if calls:
        assistant_msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}} for c in calls]
    log.info("claude cli: %.1fs cost=%s turns=%s", d.get("duration_ms", 0) / 1000, d.get("total_cost_usd"), d.get("num_turns"))
    return {"text": say, "tool_calls": calls, "assistant_msg": assistant_msg}
