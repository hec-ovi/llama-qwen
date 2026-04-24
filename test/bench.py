"""
End-to-end benchmark for llama-qwen.

Exercises the OpenAI-compatible chat endpoint with three prompt shapes:
  1. short decode    — "Hi."                    (decode-dominated)
  2. medium prompt   — 8K-token needle test     (prefill-dominated)
  3. long prompt     — 100K-token needle test   (long-context behavior)

Captures server-reported prefill / decode throughput (llama.cpp fills
`timings.prompt_per_second` and `timings.predicted_per_second`).

Rules enforced everywhere:
  - no max_tokens cap on the decode test (model decides)
  - needle tests use `/no_think` (Qwen convention) so the answer lands
    in visible output instead of inside <think>
  - thinking itself is NOT disabled globally — only for the needle
    shapes, where the task is "find the string" not "reason about it"

Writes results to test/bench_results.json and prints a Markdown table.
"""
import argparse
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

HOST = "http://127.0.0.1:8080"
MODEL = "Qwen3.6-27B-Q8_0"
NEEDLE = "IMPORTANT FACT: the secret passphrase is 'mountain-lemon-42'."
NO_THINK = "/no_think "


def post(body, timeout=1800):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{HOST}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        payload = json.loads(resp.read())
        return time.time() - t0, payload, None
    except urllib.error.HTTPError as e:
        return time.time() - t0, None, f"HTTP {e.code}: {e.read()[:400].decode(errors='replace')}"
    except Exception as e:
        return time.time() - t0, None, f"{type(e).__name__}: {e}"


def wait_ready(max_wait=300):
    for _ in range(max_wait):
        try:
            urllib.request.urlopen(f"{HOST}/health", timeout=3).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def extract(payload):
    usage = payload.get("usage", {}) or {}
    timings = payload.get("timings", {}) or {}
    msg = payload["choices"][0]["message"]
    raw = msg.get("content") or ""
    visible = raw.split("</think>", 1)[1].strip() if "</think>" in raw else raw
    return {
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or 0,
        "server_prefill_tps": round(timings.get("prompt_per_second") or 0, 2),
        "server_decode_tps": round(timings.get("predicted_per_second") or 0, 2),
        "visible": visible[:200],
    }


def run_test(name, body, runs=3):
    print(f"\n== {name} ({runs} iterations) ==")
    out = []
    for i in range(runs):
        wall, payload, err = post(body)
        if err:
            print(f"  run {i+1}: ERROR  {err[:200]}")
            out.append({"run": i + 1, "error": err, "wall": round(wall, 2)})
            continue
        stats = extract(payload)
        stats["run"] = i + 1
        stats["wall"] = round(wall, 2)
        if stats["completion_tokens"]:
            stats["decode_tps_wall"] = round(stats["completion_tokens"] / wall, 2)
        out.append(stats)
        print(f"  run {i+1}: prefill={stats['server_prefill_tps']} t/s  "
              f"decode={stats['server_decode_tps']} t/s  "
              f"{stats['completion_tokens']} tok / {wall:.1f}s wall")
    return out


def capture_mem_info():
    """GPU UMA usage via sysfs. No container-log parsing because llama.cpp
    doesn't log structured memory numbers the way vLLM does."""
    info = {}
    try:
        gtt_used = int(open("/sys/class/drm/card1/device/mem_info_gtt_used").read().strip())
        gtt_total = int(open("/sys/class/drm/card1/device/mem_info_gtt_total").read().strip())
        info["gtt_used_gib"] = round(gtt_used / 1024**3, 2)
        info["gtt_total_gib"] = round(gtt_total / 1024**3, 2)
    except Exception as e:
        info["error"] = str(e)
    return info


def build_filler(target_tokens):
    pangram = "The quick brown fox jumps over the lazy dog. "
    needed_chars = target_tokens * 4 - len(NEEDLE) - 200
    reps = max(1, needed_chars // len(pangram))
    return pangram * reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-wait", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    if not args.skip_wait and not wait_ready():
        print("server not ready")
        return

    results = {
        "model": MODEL, "host": HOST, "ts": time.time(),
        "memory": capture_mem_info(),
        "tests": {},
    }
    print(f"\n== memory info ==\n{json.dumps(results['memory'], indent=2)}")

    # Warmup (discarded)
    print("\n== warmup ==")
    post({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "temperature": 0,
    }, timeout=600)

    # 1. Short decode (thinking ON — we want a realistic user-facing shape)
    results["tests"]["short"] = {
        "label": "short decode",
        "prompt": "Explain what the Argentine peso is, in two short sentences.",
        "runs": run_test("short decode", {
            "model": MODEL,
            "messages": [{"role": "user",
                          "content": "Explain what the Argentine peso is, in two short sentences."}],
            "temperature": 0,
        }, runs=args.runs),
    }

    # 2. Medium needle ~8K (thinking OFF for a clean needle answer)
    med_prompt = NO_THINK + NEEDLE + " " + build_filler(8000) + \
                 " What is the secret passphrase? Answer with just the passphrase, nothing else."
    results["tests"]["medium_needle"] = {
        "label": "medium needle (~8K prompt)",
        "prompt": f"<needle + 8K-token filler>",
        "runs": run_test("medium needle (~8K)", {
            "model": MODEL,
            "messages": [{"role": "user", "content": med_prompt}],
            "max_tokens": 64,
            "temperature": 0,
        }, runs=args.runs),
    }

    # 3. Long needle ~100K (thinking OFF; single run — this is slow)
    long_prompt = NO_THINK + NEEDLE + " " + build_filler(100000) + \
                  " What is the secret passphrase? Answer with just the passphrase, nothing else."
    results["tests"]["long_needle"] = {
        "label": "long needle (~100K prompt)",
        "prompt": f"<needle + 100K-token filler>",
        "runs": run_test("long needle (~100K)", {
            "model": MODEL,
            "messages": [{"role": "user", "content": long_prompt}],
            "max_tokens": 64,
            "temperature": 0,
        }, runs=1),
    }

    out_path = Path(__file__).parent / "bench_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")

    # Markdown summary
    print("\n| Test | prompt_tok | completion_tok | wall (s) | prefill t/s | decode t/s |")
    print("|---|---|---|---|---|---|")
    for test in results["tests"].values():
        ok = [r for r in test["runs"] if "error" not in r]
        if not ok:
            continue
        avg = lambda k: sum(r.get(k, 0) for r in ok) / len(ok)
        print(f"| {test['label']} | {int(avg('prompt_tokens'))} | "
              f"{int(avg('completion_tokens'))} | {avg('wall'):.1f} | "
              f"{avg('server_prefill_tps'):.2f} | {avg('server_decode_tps'):.2f} |")


if __name__ == "__main__":
    main()
