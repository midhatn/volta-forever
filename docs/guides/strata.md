# Run Strata on a V100

## 1. CUDA

On Ubuntu 24.04 x86-64, run this check for the proprietary NVIDIA R580 driver.

```bash
if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | grep -E 'V100.*, 580[.]'; then
  sudo apt update && sudo apt install -y nvidia-driver-580
fi
```

If you installed the driver, reboot and repeat the check. It should print your V100 and driver version.

Strata ships the CUDA runtime and cuBLAS with its engine. The host can run this demo without installing a CUDA toolkit.

## 2. VF

Build VF 0.1.6 for the benchmark below. The build fetches its CUDA libraries into the checkout.

```sh
sudo apt update && sudo apt install -y build-essential ca-certificates curl git python3 xz-utils
if ! command -v rustup >/dev/null; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"
git clone --depth 1 https://github.com/nbald/volta-forever.git ~/volta-forever
(cd ~/volta-forever && scripts/build.sh && scripts/fetch-all.sh redist && scripts/build-forwarders.sh)
export PATH="$HOME/volta-forever/scripts:$PATH"
vf doctor
```

## 3. Engine

Install the upstream terminal, then use its signed update feed to install the CUDA 13 engine. [Strata installation](https://github.com/Niko1221/Strata#install)

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install-strata.sh
sh install-strata.sh
strata --version
STRATA_CUDA=13 vf strata update --backend cuda --no-control
```

## 4. Test

Download the 22.63 GB UD-Q4_K_M GGUF. Strata uses the catalog profile `qwen36` and the size preset `UD-Q4_K_XL`. The explicit `--model-path` selects the verified Qwen3.5 UD-Q4_K_M file. [Checkpoint](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF)

```sh
model=Qwen3.5-35B-A3B-UD-Q4_K_M.gguf
curl -fL --retry 3 -o "$model" \
  "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF/resolve/63af8373893a7a73c6dfcb84cb63d815981da5e0/$model" &&
printf '%s  %s\n' a20c5a9f0cee94d24af3553e7828df1337c7d24213789911c74e1f4936894936 "$model" | sha256sum --check &&
CUDA_MODULE_LOADING=LAZY STRATA_CUDA=13 vf strata launch --family qwen36 --model UD-Q4_K_XL \
  --model-path "$PWD/$model" --context 8k --parallel 1 --vision off --drafting off --mtp off --tune skip --set STRATA_Q36_BLAS_F32=1 \
  --set prompt-cache=0 --set conversation-cache-mib=0 --access local --port 9085 \
  --no-api-key --yes --no-tui
```

Ask “What is the capital of France? Answer in one word.” in the local chat. The unpinned installation selected the signed Strata 0.2.18 terminal and CUDA 13 engine 0.1.64. It answered **Paris** on a Tesla V100-SXM2-32GB with driver 580.178.04 with the CUDA 12 native engine and with CUDA 13 under VF.

### Several GPUs

With the layers split over several GPUs (`--gpu 0,1`), the engine captures its first verify window inside its first request, under its watchdog of 60 s, and the kernels of that window are adapted and compiled there: the first start spends tens of seconds in that request. On two V100-SXM2-32GB with two Xeon Gold 6133 and empty caches, the first window took 33 s and 43 s in two runs with VF from main on 2026-11-23, before 0.1.4, which has that change, most of it on one CPU core, so a slower machine has little margin; with VF 0.1.3 it took 212 s, and the watchdog stopped the engine (issue #3). For the first start, give the watchdog more time in the environment of the launch:

```sh
STRATA_WATCHDOG_S=900 CUDA_MODULE_LOADING=LAZY STRATA_CUDA=13 vf strata launch ... --gpu 0,1 ...
```

VF and the driver keep what they made in their caches (here 7 MB and 15 MB; with VF 0.1.3 the driver's took 140 MB, which `CUDA_CACHE_MAXSIZE` must allow), and the next starts need no setting: the first window then took 1.7 s and 2.9 s in two runs (31 s with VF 0.1.3). Another build of VF does not use the layer's cache of the one before: give the watchdog the same time at the first start after an upgrade.

### Engine 0.1.66 and long prompts

Strata 0.2.20 installs the CUDA 13 engine 0.1.66. When it detects VF, that engine maps the 7.5 that VF reports back to 7.0 (its log says `compute capability 7.0 (volta-forever 0.1.4: remapped from reported 7.5)`), and then runs the prompt chunks that go to the GPU on its int8 expert kernels for Volta (`i8h_kernel`). Its CUDA 13 build has no code for those kernels: their body is a trap, for every architecture of that build (its CUDA 12 build has sm_70 code for them). With VF 0.1.4 the first prompt of a few thousand tokens (4,021 here) stops the GPU: the engine ends with `native_expert_i8/down: unspecified launch failure`, the system log (`dmesg`) gets an Xid 43 line, and the server answers 503 (issue #3). With VF 0.1.5 the layer refuses that launch, with a line that names the kernel (`kernel ...i8h_kernel... has no code in its image ...: its body is a trap and nothing else; cuLaunchKernel with it is refused (error 218)`), and the engine ends with `native_expert_i8/gu: a PTX JIT compilation failed`, without a GPU fault; the server answers 503 as well. Set `STRATA_I8_SM70=0` in the environment of the launch:

```sh
STRATA_I8_SM70=0 CUDA_MODULE_LOADING=LAZY STRATA_CUDA=13 vf strata launch ...
```

The engine then still says `every chunk runs 48 of 48 layers on the int8 kernels`, and asks for its other int8 kernels (`i8_kernel`) instead. `STRATA_EXPERT_I8=0`, which the report found, takes the FP16 path, and reads prompts more slowly. Measured on two V100-SXM2-32GB (driver 580.178.04, two Xeon Gold 6133) with the IQ3_S checkpoint and the flags of issue #3 (`--gpu 0,1 --context 64k --parallel 1 --vision off --drafting off --mtp off --tune skip --set prompt-cache=0 --set conversation-cache-mib=0`), VF from main on 2026-11-24, before 0.1.5: each prompt was read twice in one server, with nothing reused, and 48 tokens generated at temperature zero. The figures are the engine's, for the second read (the first includes the compilation of the kernels that a prompt size uses first).

| Prompt (tokens) | CUDA 12 engine, no VF: read / decode (tokens/s) | CUDA 13 engine with VF, `STRATA_I8_SM70=0` | with VF, `STRATA_EXPERT_I8=0` |
| ---: | ---: | ---: | ---: |
| 4,021 | 1,510 / 70.1 | 1,385 / 64.5 | 1,076 / 64.5 |
| 8,102 | 1,637 / 68.7 | 1,504 / 58.8 | 1,427 / 58.6 |
| 16,270 | 2,225 / 68.3 | 2,039 / 58.3 | 1,945 / 57.5 |
| 35,124 | 2,729 / 66.6 | 2,498 / 57.0 | 2,272 / 57.4 |

With either setting, the replies to all nine requests (a short question, then the four prompts twice) were the native engine's text. With empty caches the first window took 32.0 s at the default settings and 32.1 s with each setting.

That build also has kernels whose body is empty for every architecture, where its CUDA 12 build has sm_70 code: `q8_v70_kernel` (21 instances) and `gr_up_ts_kernel` and `gr_down_ts_kernel` (16). Launched, such a kernel does nothing and returns no error, so VF does not refuse it, and the engine's results would be wrong without a message. In these runs the engine never asked for any of them: VF compiles each kernel of these libraries when the engine first asks for it, and names it then at `VF_LOG_LEVEL=2` (or with `VF_TIMING=1`), and none of these was named, at the default settings up to the refused launch, and with either setting up to the prompt of 35,124 tokens. `VF_LOG_LEVEL=2` also says, for each image that it adapts, how many of its kernels have a body that is a trap and how many an empty one.

## 5. Benchmarks

With that server running, download the client and prompts to the same directory. The client runs a warm-up and three different chats at temperature zero with thinking disabled.

```sh
base=https://raw.githubusercontent.com/nbald/volta-forever/main/docs/guides/benchmarks
curl -fsSLO "$base/strata-ninfer.py"
curl -fsSLO "$base/prompts.json"
python3 strata-ninfer.py strata
```

**MTP and all other drafting were disabled** (`--drafting off --mtp off`). Each of three distinct requests read 227 uncached tokens and generated 64. The replies began with Kyoto, Lyon and Lima. The server reported zero draft proposals and zero acceptances on every request. The decode rate counts the 63 steps after the first token.

| On the same V100-SXM2-32GB, same checkpoint and cuBLAS setting | Median prefill (tokens/s) | Median decode (tokens/s) |
| --- | ---: | ---: |
| Strata CUDA 13 with VF from main | **2000.00** | **109.13** |
| Native Strata CUDA 12.9 | 1968.80 | 105.19 |

Both engines passed the same four prompts on the same V100. [Per-request timings and replies](./benchmarks/common-gguf-q4-k-m.json)

To repeat the native comparison, stop the VF server, select the CUDA 12 engine, and repeat section 4 with `STRATA_CUDA=12 strata launch`. Keep `--model-path` pointed at the GGUF verified in section 4 and keep all other launch options the same.

```sh
strata launch --stop --no-tui --yes
STRATA_CUDA=12 strata update --backend cuda --no-control
```

## 6. Additional notes

The standard Q4_K_M from the separate Unsloth repository lacks `qwen35moe.nextn_predict_layers`. Both Strata builds required this metadata even with MTP disabled. The UD-Q4_K_M file used here also runs in llama.cpp and FreeToken. [Common checkpoint and measurements](./benchmarks/common-gguf-q4-k-m.json)
