#!/usr/bin/env python3
"""Real-time viewer for claude -p --output-format stream-json.

Reads JSON lines from stdin, prints human-readable summary of what Claude is doing.
Usage: claude -p "..." --output-format stream-json 2>&1 | tee log.jsonl | python3 stream_viewer.py
"""
import sys
import json

GREY = "\033[90m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"

def truncate(s, n=200):
    s = s.replace("\n", " ")
    return s[:n] + "..." if len(s) > n else s

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue

    msg_type = d.get("type")

    # Assistant message (thinking, text, tool_use)
    if msg_type == "assistant":
        content = d.get("message", {}).get("content", [])
        if isinstance(content, str):
            print(f"{GREEN}[REPLY]{RESET} {truncate(content)}")
            continue
        for block in content:
            bt = block.get("type")
            if bt == "thinking":
                text = block.get("thinking", "")
                print(f"{GREY}[THINK] {truncate(text, 150)}{RESET}")
            elif bt == "text":
                text = block.get("text", "")
                print(f"{GREEN}[TEXT]{RESET} {truncate(text, 300)}")
            elif bt == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input", {})
                # Show key info per tool type
                if name == "Bash":
                    detail = inp.get("command", "")[:150]
                elif name in ("Read", "Write"):
                    detail = inp.get("file_path", "")
                elif name == "Edit":
                    detail = inp.get("file_path", "")
                elif name == "Grep":
                    detail = f"/{inp.get('pattern', '')}/ in {inp.get('path', '.')}"
                elif name == "Glob":
                    detail = inp.get("pattern", "")
                elif name == "Agent":
                    detail = inp.get("description", "")
                else:
                    detail = str(inp)[:100]
                print(f"{CYAN}[TOOL] {name}{RESET} → {truncate(detail, 150)}")

    # Tool result
    elif msg_type == "tool_result" or msg_type == "user":
        content = d.get("message", {}).get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    text = block.get("content", "")
                    if isinstance(text, list):
                        text = " ".join(b.get("text", "") for b in text if isinstance(b, dict))
                    is_err = block.get("is_error", False)
                    color = RED if is_err else YELLOW
                    label = "ERROR" if is_err else "RESULT"
                    print(f"{color}[{label}]{RESET} {truncate(str(text), 200)}")

    # Result (final)
    elif msg_type == "result":
        text = d.get("result", "")
        cost = d.get("cost_usd", 0)
        turns = d.get("num_turns", 0)
        duration = d.get("duration_ms", 0) / 1000
        print(f"\n{BOLD}{'='*50}")
        print(f"DONE — {turns} turns, {duration:.0f}s, ${cost:.2f}")
        print(f"{'='*50}{RESET}")
        if text:
            print(truncate(text, 500))
