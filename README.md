<h1 align="center">llama-qwen</h1>

<p align="center">
  <strong>Qwen3.6-27B (Q8_0 GGUF) served over OpenAI-compatible HTTP on AMD Strix Halo.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Working-brightgreen" alt="Status" />
  <img src="https://img.shields.io/badge/Model-Qwen3.6--27B-0b7285" alt="Model" />
  <img src="https://img.shields.io/badge/Quant-Q8__0_GGUF-purple" alt="Quant" />
  <img src="https://img.shields.io/badge/Context-256K_native-orange" alt="Context" />
  <img src="https://img.shields.io/badge/Decode-7.5_t%2Fs-blue" alt="Decode" />
  <img src="https://img.shields.io/badge/Prefill-200_t%2Fs-blue" alt="Prefill" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Ubuntu-26.04-E95420?logo=ubuntu&logoColor=white" alt="Ubuntu" />
  <img src="https://img.shields.io/badge/ROCm-TheRock_7.13-ED1C24?logo=amd&logoColor=white" alt="ROCm" />
  <img src="https://img.shields.io/badge/GPU-gfx1151_(RDNA_3.5)-ED1C24?logo=amd&logoColor=white" alt="GPU" />
  <img src="https://img.shields.io/badge/llama.cpp-upstream_HEAD-4B2E83" alt="llama.cpp" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker" />
</p>

---

## What this is

A thin Docker Compose wrapper that runs `unsloth/Qwen3.6-27B-GGUF` (Q8_0)
behind an OpenAI-compatible HTTP API on an **AMD Ryzen AI Max+ 395
"Strix Halo"** (`gfx1151`, RDNA 3.5, 128 GB UMA). Serves
`/v1/chat/completions` and `/v1/completions` through the same
endpoint. Native 256K context with **quantized KV cache** (q8_0)
to keep the cache small enough that the full context still fits.

llama.cpp is **built from source** against a TheRock nightly ROCm SDK
for gfx1151, with rocWMMA flash-attention kernels enabled. No patches,
no forks — upstream llama.cpp works as-is for this architecture, which
is a key difference from the vLLM path.

This is the **sibling** of [`vllm-qwen`](https://github.com/hec-ovi/vllm-qwen).
Same hardware, same model family, different runtime. See the
[comparison](#which-should-i-run) below.

---

## Stack

| Layer | Version |
|---|---|
| Host OS | Ubuntu 26.04 (container base) |
| ROCm | **TheRock `7.13.0a20260424`** (S3 nightly; resolves to latest at build time) |
| llama.cpp | upstream HEAD (build pinned to `15fa3c493` — 3 commits behind at bench time, all unrelated to HIP/gfx1151) |
| Model | `unsloth/Qwen3.6-27B-GGUF`, Q8_0 quant |
| Build flags | `GGML_HIP=ON` + `GGML_HIP_ROCWMMA_FATTN=ON` + `GPU_TARGETS=gfx1151` |

---

## Hardware

Tested on: **Ryzen AI Max+ 395 / 128 GB UMA** (Radeon 8060S iGPU, `gfx1151`).
Kernel ≥ 6.18. Docker with `/dev/kfd` + `/dev/dri` access.

This is **the only supported configuration today.** Other Strix Halo
variants (8050S / 8040S / lower RAM) will likely work but haven't been
tested.

### Host memory setup (required — one-time)

Strix Halo is UMA: system RAM and GPU VRAM share the same physical pool.
Out of the box the BIOS reserves a fixed chunk as "dedicated VRAM" and
the Linux TTM subsystem caps how much of the rest the GPU driver may
map as GTT. Both defaults are wrong for this workload — the 27 GiB
Q8_0 model plus a 256K KV cache won't fit unless you widen them.

**1. BIOS / UEFI:** set the dedicated GPU VRAM carve-out to its
**minimum (2 GB / 2048 MB)**. You want the GPU to take memory from the
shared pool on demand via GTT, not from a fixed-size pre-allocation.
Menu name varies by vendor — look for *UMA Frame Buffer Size*,
*UMA Buffer Size*, *iGPU Memory*, or *GPU Shared Memory*.

**2. Ubuntu GRUB:** raise the TTM page limit so the kernel will
actually let the GPU driver map ~116 GiB of GTT. Edit
`/etc/default/grub`, set:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash ttm.pages_limit=30408704 amdgpu.noretry=0 amdgpu.gpu_recovery=1"
```

Then:

```bash
sudo update-grub
sudo reboot
```

Verify after reboot:

```bash
cat /sys/class/drm/card1/device/mem_info_gtt_total
# expect ~124554670080  (≈ 116 GiB)
```

`ttm.pages_limit=30408704` is 30,408,704 × 4 KiB pages = **116 GiB**.
Leave 12 GiB for the OS and desktop. `amdgpu.noretry=0` +
`amdgpu.gpu_recovery=1` are stability flags — keep them on for
long-running inference.

---

## Quick start

```bash
cp .env.template .env
# edit .env:
#   - LLAMA_HOST_MODELS_DIR: your HF cache directory
#   - HF_TOKEN: optional (the Unsloth GGUF repo is public, not gated)

# Fetch the Q8_0 GGUF into your cache directory
hf download unsloth/Qwen3.6-27B-GGUF \
  --include "*Q8_0*" \
  --cache-dir "$LLAMA_HOST_MODELS_DIR/hub"

# Confirm the snapshot path `hf download` created, then update
# LLAMA_MODEL_FILE in .env to point at the actual file. The sha in the
# template default is pinned to the version used at bench time and
# won't match your snapshot dir.

docker compose up -d --build

# First build: ~10 min (ROCm tarball + llama.cpp HIP source build, -j 4)
# First start: <15s (no JIT — llama.cpp compiles all kernels ahead of time)
# Subsequent starts: <10s

# Verify
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool
```

---

## First boot

Unlike the vLLM build, llama.cpp has **no Triton JIT phase**. Every
HIP kernel is compiled once at Docker-build time (the ~10 min CMake
step), then the runtime just `mmap`s the `.so`. Cold-start timing:

| Phase | Time | Visible activity |
|---|---|---|
| Container init | ~2s | dynamic linker |
| Weight load | ~7s | 28 GiB Q8_0 streaming into UMA (no mmap → full prefetch) |
| KV alloc + warm | ~3s | allocate q8_0 cache for 256K slots |
| Server bind | ~1s | HTTP server listens on `:8080` |
| **Total** | **~13s** | |

No "silent window" to misread as stuck — llama-server logs its load
progress continuously. You can issue the first request the moment
`/health` returns 200.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat with `messages`. Supports streaming (`stream: true`). |
| `POST /v1/completions` | OpenAI text completion — raw `prompt` string. |
| `GET  /v1/models` | Lists the loaded GGUF. |
| `GET  /health` | Liveness probe. |
| `GET  /props` | llama.cpp-specific: loaded model metadata, context size, chat template. |

The model's native **thinking mode** (`<think>...</think>`) is on by
default. Add `/no_think` anywhere in the user message to suppress it
for a single turn (Qwen convention) — the benchmark's needle tests
use this so the answer lands in visible output instead of inside
`<think>`.

**No `/v1/responses`** and **no vision input**. llama.cpp doesn't
implement the Responses API, and the Unsloth GGUF is text-only (the
vision tower isn't included in the GGUF conversion). For either, use
[`vllm-qwen`](https://github.com/hec-ovi/vllm-qwen).

---

## Benchmark

Hardware: Ryzen AI Max+ 395, 128 GB UMA, Ubuntu 26.04, TheRock 7.13 nightly.
Model: `unsloth/Qwen3.6-27B-GGUF` Q8_0, 256K context with q8_0 KV.
Warmup once, then numbers averaged over 3 iterations. Temperature 0.
Server-reported `timings.prompt_per_second` and
`timings.predicted_per_second` are the canonical throughput fields.

| Shape | prompt_tok | completion_tok | wall (s) | prefill t/s | decode t/s |
|---|---|---|---|---|---|
| Short decode ("Hi.") | 12 | 202 | 27.1 | 42.4 | **7.55** |
| Medium needle (~8K prompt) | 8,039 | 198 | 66.9 | 200.0 | **7.43** |

Decode is **rock-steady at 7.4–7.6 t/s** across configs. That's the
real Q8_0 ceiling for a 27B model on gfx1151 — bound by weight-streaming
bandwidth through the UMA, not by compute.

### Decode throughput across tuning attempts

I swept a handful of obvious knobs. None of them moved the needle —
decode is pure memory bandwidth on this GPU:

| Config | Short decode (t/s) | Medium prefill (t/s) |
|---|---|---|
| baseline (defaults) | 7.53 | 196.91 |
| `-t 16 -tb 32` (physical cores + full SMT) | 7.56 | 199.97 |
| `-b 4096 -ub 1024` (bigger prefill batches) | 7.48 | 188.83 |
| `GGML_HIP_UMA=1` (unified memory mode) | 7.55 | 199.17 |

Bigger prefill batches actually *hurt* prefill (197 → 188 t/s) — the
kernel already saturates at its default sizes, and the extra
synchronization cost dominates.

### Decode vs. vLLM BF16 on the same hardware

| Runtime | Precision | Weights on disk | Decode t/s |
|---|---|---|---|
| vLLM | BF16 | 51.2 GiB | 4.30 |
| **llama.cpp** | **Q8_0** | **~27 GiB** | **7.50** |

Q8 is ~60% the weight bandwidth of BF16, so decode is ~75% faster in
practice. That is the only material reason to prefer this path —
quality-wise the Q8 output is essentially indistinguishable from BF16
for everyday use.

### Functional verification

Spot-checked the endpoints on the current build to confirm behavior:

| Endpoint / feature | Result |
|---|---|
| `/v1/chat/completions` with thinking ON (300-tok cap) | decode **7.5 t/s**, prefill 53 t/s on 23-tok prompt |
| `/v1/completions` ("The capital of Argentina is") | decode **7.9 t/s**, returns `" Buenos Aires, which is…"` |
| `/v1/chat/completions` single tool call (`get_weather`, Tokyo) | `finish_reason=tool_calls`, clean `{"city":"Tokyo"}`, empty content |
| `/v1/chat/completions` parallel tool calls (Tokyo + Rosario) | 2 structured calls, zero content leakage |
| `/v1/chat/completions` tool call with optional arg | clean `{"city":"Rosario, Argentina","unit":"celsius"}` |
| `/v1/chat/completions` with `image_url` | HTTP 500: `image input is not supported - hint: if this is unexpected, you may need to provide the mmproj` — confirms the GGUF has no vision tower |

Tool-call parsing on this path is **not** affected by the vLLM
reasoning/tool-parser bugs ([vllm#40783](https://github.com/vllm-project/vllm/pull/40783),
[#40785](https://github.com/vllm-project/vllm/pull/40785),
[#40787](https://github.com/vllm-project/vllm/pull/40787)) — llama.cpp
has its own independent extractor.

---

## Reproduce

```bash
python3 test/bench.py
# writes test/bench_results.json with full per-run detail
```

`test/bench.py` warms up once, then runs the three prompt shapes and
records server-reported `timings.*` counters. No external deps beyond
Python 3.

---

## Which should I run: `llama-qwen` or `vllm-qwen`?

| | `llama-qwen` (this repo) | [`vllm-qwen`](https://github.com/hec-ovi/vllm-qwen) |
|---|---|---|
| Weights format | Q8_0 GGUF | BF16 safetensors (official) |
| Decode speed | **~7.5 t/s** | 4.3 t/s |
| Prefill speed (8K) | 200 t/s | ~38 t/s |
| Vision input | ✘ | ✓ |
| `/v1/responses` + separated reasoning | ✘ | ✓ |
| Tool calling (OpenAI format) | ✓ (via `--jinja`, verified clean) | ⚠ broken on current vLLM commit (see [vllm#40783](https://github.com/vllm-project/vllm/pull/40783) / [#40785](https://github.com/vllm-project/vllm/pull/40785) / [#40787](https://github.com/vllm-project/vllm/pull/40787)) |
| Context | 256K | 256K |
| Memory footprint | ~35 GiB total | ~105 GiB total |
| Boot time cold | **~13s** | ~4 min |
| Build from source needs patches | no | yes (12 local patches) |
| Official weights | ✘ (Unsloth re-quant) | ✓ (Qwen BF16) |

**Rule of thumb:** if you need raw speed, agentic loops, or a fast
desktop sidekick → `llama-qwen`. If you need vision, reasoning
separated for structured pipelines, or weights directly from the
Qwen team → `vllm-qwen`.

---

## Known limitations

| Target | Status | Root cause |
|---|---|---|
| Vision / `image_url` | ✘ unsupported | Unsloth's GGUF conversion doesn't include the Qwen 3.6 vision tower. Fixable upstream in the GGUF export path, not here. |
| `/v1/responses` with separated reasoning | ✘ unsupported | llama.cpp doesn't implement the OpenAI Responses API. |
| FP8 KV cache | n/a | RDNA 3.5 has no hardware FP8 path — Q8_0 is already the correct quant for this GPU. |

---

## Repo layout

```
.
├── Dockerfile              multi-stage: Ubuntu + TheRock + llama.cpp HIP source build
├── docker-compose.yml      one service, one model, host-mounted cache
├── .env.template           the one config file you need to edit
├── scripts/
│   └── install_rocm_sdk.sh TheRock S3 nightly tarball → /opt/rocm
└── test/
    └── bench.py            reproducible 3-shape benchmark harness
```
