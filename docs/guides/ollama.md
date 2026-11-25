# Ollama on a Tesla V100

Every measurement on this page is from an outside tester, TheWeakShall, in
[issue #4](https://github.com/nbald/volta-forever/issues/4): Ollama 0.40.4 and VF 0.1.5 on a
V100-SXM2-32GB, with drivers 580.173.02 and 580.178.04; the end of section 6 is a run with the
code of 0.1.6 (`e3c870b`). The project did not run Ollama itself;
[docs/reports/](../reports/README.md) has a row for each report, with its host and its link.

## 1. What stock Ollama ships

The Linux x86-64 archive of Ollama 0.40.4 carries two CUDA runners. `lib/ollama/cuda_v12` has
machine code for several architectures, sm_70 among them; `lib/ollama/cuda_v13` has PTX for
compute capability 7.5 and later only. Without VF, Ollama skips `cuda_v13` on a V100
(`skipping CUDA device — compute capability not in compiled architectures`) and runs the model
on `cuda_v12`, at compute 7.0. A V100 needs no VF for that runner.

Through `vf`, the `cuda_v13` runner is told 7.5 and accepts the V100: its kernels are its own
PTX, retargeted to the V100 by VF, and the cuBLAS it asks for (`libcublas.so.13`) is VF's
forwarding library to cuBLAS 12.9. That is the case measured below.

## 2. VF

```sh
curl -fsSL https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
vf doctor
```

`vf doctor` says whether the layer can use its cache directory. With VF 0.1.5, the tester's first
cache directory, made under a umask of 002, was writable by its group, and the layer turned its
cache off with one line in its log: every start then adapted the runner's code again (a warm start
of 18.3 s instead of 11.0). `chmod 700 DIR` on that directory is the fix.

## 3. Start the server

Run the server by hand as your own user, not as Ollama's system service, which runs as another
user, with another home and other caches, and would need `vf` in its command. The tester chose
the CUDA 13 runner with `OLLAMA_LLM_LIBRARY`:

```sh
OLLAMA_LLM_LIBRARY=cuda_v13 VF_LOG=$PWD/vf.log vf ollama serve
```

The server's `inference compute` line then says `compute=7.5` and `libdirs=ollama,cuda_v13`;
`compute=7.0` with `cuda_v12` is the CUDA 12 runner, without the layer's work. While a model is
loaded, the libraries of the runner process:

```sh
grep -E 'libcuda|cublas' /proc/$(pgrep -n -f llama-server)/maps | awk '{print $6}' | sort -u
```

Among them are VF's `lib/libcuda.so.1`, NVIDIA's `libcuda.so.580...`, and VF's
`forward/libcublas.so.13` and `forward/libcublasLt.so.13`; Ollama's own `cuda_v13/libcublas.so.13`
should not be. `vf.log` has the layer's lines: in every run of the tester, kernels with emulated
instructions (`emulated in ...`), and no kernel refused.

The measurements below were made with `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_CONTEXT_LENGTH=32768`, and
`OLLAMA_FLASH_ATTENTION=1` (and `0` for the first model), greedy requests (`temperature` 0,
`top_k` 1, `seed` 1) in raw mode (the vision model with its chat template), each with
`prompt_eval_cached_count` 0 (Ollama reuses a prompt it has seen).

## 4. Speed

Three rounds, each a new server start with a warm-up first, the runners alternated. The figures
are the ratios of the report's medians: the CUDA 13 runner through `vf` against the stock CUDA 12
runner, on three short prompts and two long ones (about 4,000 and 16,000 tokens).

| Model | Prompts | Generation |
|---|---:|---:|
| Qwen3.5-35B-A3B UD-Q4_K_M (mixture of experts), flash attention on | 1.58 to 1.90 | 0.98 to 1.02 |
| the same, flash attention off | 1.50 to 1.68 | 0.98 to 1.00 |
| granite4:tiny-h Q4_K_M (hybrid, with mixture-of-experts layers) | 1.26 to 1.84 | 0.96 to 1.01 |
| qwen3:8b Q8_0 (dense) | 0.43 to 0.52 | 0.92 to 1.00 |
| qwen3-vl:8b Q4_K_M (dense, with an image prompt among its six) | 0.40 to 0.67 | 0.93 to 1.01 |

Told 7.5, llama.cpp (which Ollama's runners are built on) computes every quantized prompt batch
with its MMQ kernels, whose integer tensor-core products VF emulates on Volta. Built for sm_70, it
sends batches of 64 rows or more to cuBLAS on the V100's fp16 tensor cores: faster on dense
layers, slower on mixtures of experts, where it calls cuBLAS expert by expert (`LIMITS.md`,
"Speed"). So for a dense model the stock `cuda_v12` runner is the faster choice, and it needs no
VF. A source build of the same Ollama for sm_70 gave the stock `cuda_v12` runner's text token for
token, at about the same speed.

The first request of a new prompt length is slower while its kernels compile: 592 tokens/s in the
first round for the longest prompt without flash attention, 975 after. A first start with the
layer's and the driver's caches empty (`VF_PROXY_CACHE` and `CUDA_CACHE_PATH` new directories,
`CUDA_CACHE_MAXSIZE=4294967296`) answered a 32-token request in 43.0 s (`load_duration` 25.4 s),
the next start with the same caches in 11.0 s (10.2 s), the model's file in the page cache both
times; the caches then held 3.8 MB and 5.2 MB. The `cuda_v12` runner loaded in 6 to 11 s.

## 5. Text

Each runner gave the same text in every round. With Qwen3.5-35B-A3B, wherever the CUDA 13
runner's text first left the CUDA 12 runner's, the CUDA 12 runner's first two candidates were at
most 0.27 apart in log-probability, and its choice was among the CUDA 13 runner's first three:
different kernels for the same arithmetic. granite4:tiny-h's log-probabilities moved about
three times as far as the other models', and its text left the native text twice where the native
runner led by 0.57 and 0.63; llama.cpp itself takes other ways at 7.5 for its hybrid layers
(`LIMITS.md`, "Numerical accuracy").

With greedy decoding such a near tie can also send a text into a repetition loop, which Ollama's
repeat limit ends with HTTP 500 (`prediction aborted, token repeat limit reached`): 2 of the
4,000 requests of the long run below, which complete with the CUDA 12 runner and loop through the
layer, again on a new server; a third loops with both.

One flash-attention case of llama.cpp b11521, the version Ollama 0.40.4 has (head size 128, eight
query heads to a KV head, a KV length of 5,003, four rows), aborts on both sm_70 builds, the
stock `cuda_v12` runner's and a source build's (`flash_attn_ext_f16 has no device code compatible
with CUDA arch 700`), with an Xid 43 line; through `vf`, the `cuda_v13` runner's library gives the
CPU's result (`test-backend-ops`). It is an upstream bug (llama.cpp issue 31555), and
`tests/gpu/llamacpp-ops.sh` runs the case through the layer only.

## 6. Several runners, MPS, a long run

- **Two runner processes** (`OLLAMA_MAX_LOADED_MODELS=2`, Qwen3.5-35B-A3B and granite4:micro,
  both requests at the same instant, both caches empty): both runners adapted, compiled and wrote
  the layer's and the driver's caches at once, without a line about them, each reply was the text
  of the model alone, and the next server used the caches (loads of 23.8 and 29.9 s, then 9.9
  and 2.8 s).
- **MPS**: the same two models as clients of one MPS server (`nvidia-cuda-mps-control -d`; both
  runners `M+C` in `nvidia-smi`), on driver 580.178.04: the same text and the same
  log-probabilities as without MPS, with empty caches and with warm ones; no Xid line.
- **A long run**: 4,000 greedy requests of 64 to about 12,000 prompt tokens to Qwen3.5-35B-A3B
  (`num_ctx` 16384) in 4 h 10 min, one runner process throughout. Nothing drifted from the first
  hour to the last: the runner's memory (once Ollama's prompt cache filled), the GPU's, the
  layer's cache (33.0 and 33.2 MB) and the driver's (44.6 and 44.8 MB); generation at 95.9 and
  96.4 tokens/s, prompts at 1,510 and 1,494 (medians of the first and the last thousand
  requests); no Xid line, no CUDA error. Two fixed prompts kept their text through 80 rounds; the
  third changed where Ollama began to reuse one token of it, and the CUDA 12 runner's changed at
  the same places in a replay of the first 1,950 requests. That replay, with the same reuse of
  prompts on both runners, gave by new prompt tokens:

  | New prompt tokens | Prompts, `cuda_v12` / `cuda_v13` through `vf` (tokens/s) | Generation (tokens/s) |
  |---|---:|---:|
  | under 512 | 429 / 490 | 98.9 / 98.9 |
  | 512 to 2,048 | 826 / 1,507 | 98.3 / 97.8 |
  | 2,048 to 6,000 | 887 / 1,573 | 95.2 / 94.5 |
  | 6,000 to 13,000 | 901 / 1,548 | 94.1 / 91.7 |

  With the code of 0.1.6 (`e3c870b`), on a new server, the three fixed prompts gave the same bytes
  as with 0.1.5 and as in the long run's first round, and the three requests that loop (section 5)
  ended with HTTP 500 again.

On four more systems with driver 580.178.04 (Rocky Linux 9.9, Ubuntu 26.04, Debian 13, Fedora
45), `qwen3:0.6b` through `vf` gave the CUDA 12 runner's text, and the server said
`compute=7.5 libdirs=ollama,cuda_v13`.
