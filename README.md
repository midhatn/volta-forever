# volta-forever

**Your Tesla V100 is not dead.**\
CUDA 13 dropped Volta.\
volta-forever brings it back.

Run unmodified CUDA 13 applications on a Tesla V100 with a simple two-letter command prefix: `vf`.

[Inference engines and Ubuntu guides](#inference-engines) · [Installation](INSTALL.md) · [Compatibility limits](LIMITS.md)

> **A young project.** volta-forever runs the engines listed below and a large part of PyTorch,
> but not every CUDA 13 program works out of the box yet: a kernel may be refused, a library may
> need a CUDA 12 counterpart, a program may need a setting. Read [the limits of 0.1.5 to know
> first](LIMITS.md#the-limits-of-015-to-know-first). The recommended way to get a program
> running is to hand the job to your AI coding agent: point it at this repository, tell it which
> program you want on your V100, and ask it to follow [AGENTS.md](AGENTS.md). When the layer
> itself is what is missing, ask the agent to fix it here and to send a pull request: that is
> how the project grows. [Reporting a wrong result](CONTRIBUTING.md#reporting-a-wrong-result)

> **Triton: validated on a corpus, and slow at the default capability.** Triton generates its
> kernels while a program runs, for that program's shapes and settings, and for a program told
> 7.5 (the default) they run from their machine code. Of 3,516 configurations of 20 Triton
> kernels and 241 cases of vLLM's and SGLang's Triton kernels, every one that Triton compiles
> gave on a V100 what Triton's interpreter computes, where it computes it, and what its own PTX
> gives, and the independent validator found no problem in the 4,308 cubins of the first
> ([validation](docs/validation.md)). A kernel that a program makes with other shapes or settings
> was checked by neither: compare results with a reference and
> [report differences](CONTRIBUTING.md#reporting-a-wrong-result). Told 7.5, Triton 3.8 uses no
> tensor cores for `tl.dot`: its float16 and bf16 products ran 3 to 150 times slower than told
> 8.0 ([speed](LIMITS.md#speed)). In a program with PyTorch, only
> `VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1` changes that in 0.1.5, and PyTorch's own attention is
> then refused ([capability](LIMITS.md#the-capability-told-with-vf_capability)).

## Quick start

You need a Tesla V100, the NVIDIA R580 driver, Linux x86-64 with glibc 2.34 or later, and `curl`, `python3` and `xz`. Ubuntu 22.04+ and Debian 12+ meet the glibc requirement. The installer places `vf` in `~/.local/bin` and downloads NVIDIA’s CUDA 12.9.2 cuBLAS and nvJitLink.

```sh
curl -fsSL https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
vf doctor
```

Run an existing CUDA program, or use `vf venv` to install ordinary CUDA wheels.

```sh
vf llama-completion -m model.gguf -p "Hello" -n 48 -ngl 99

vf venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name())"
```

Run `vf fetch --all` if the application needs more CUDA 12.9 libraries. PyTorch on several GPUs is best served by NVIDIA’s CUDA 12 NCCL: `vf fetch --only libnccl` downloads it (the CUDA 13 NCCL of the wheels runs too since 0.1.6, after a first start that compiles it for minutes: `LIMITS.md`). [Installation and launcher options](INSTALL.md) · [Several GPUs](LIMITS.md#several-gpus-and-nccl)

## How it works

CUDA 13 applications ship kernels for GPUs newer than Volta. VF intercepts module loads, retargets supported PTX and translates supported Turing (`sm_75`) machine code to Volta (`sm_70`). The NVIDIA driver runs the resulting kernels on the V100. VF caches translations and routes libraries such as cuBLAS to their CUDA 12.9 implementations.

CUDA 13 programs see capability 7.5 by default. Applications with native Volta kernels keep 7.0. `VF_CAPABILITY=8.0 vf ./my_program` tells an application 8.0, so that it takes its Ampere PTX, as the engine guides below do. An application whose kernels VF can only give from their `sm_75` machine code, such as PyTorch, or from their `sm_75` PTX, is still told 7.5, and VF says so. VF can emulate instructions used by BF16, FP8 and NVFP4 kernels. VF also uses PTX retained inside a cubin and translates selected `sm_80` machine code; that translation is not checked by the independent validator. When a kernel needs an unsupported instruction, VF reports the kernel name and the reason. [Architecture](docs/architecture.md) · [Instruction emulation](docs/emulations.md) · [Supported images and limits](LIMITS.md)

## Inference engines

All eight engines ran **Qwen3.5-35B-A3B** on a Tesla V100-SXM2-32GB with **MTP and all other speculative decoding disabled**. Each engine processed one warm-up followed by three distinct 227-token prompts at concurrency 1 and generated 64 tokens per prompt. Rates are medians in tokens/s. Each native/VF pair used the same card and checkpoint. Strata, FreeToken and llama.cpp shared the same GGUF file.

| Engine (Ubuntu guide) | Tested model / quantization | CUDA 12 prefill | CUDA 12 decode | CUDA 13 + VF prefill | CUDA 13 + VF decode |
|---|---|---:|---:|---:|---:|
| [vLLM](docs/guides/vllm.md) | QuantTrio AWQ W4A16 | 1,443.33¹ | 76.09¹ | 1,449.98 | 80.23 |
| [SGLang](docs/guides/sglang.md) | QuantTrio AWQ W4A16 | Can't run | Can't run | 2,010.01 | 79.55 |
| [llama.cpp](docs/guides/llamacpp.md) | Unsloth GGUF UD-Q4_K_M | 494.57 | 101.27 | 1,089.57 | 101.11 |
| [FreeToken](docs/guides/freetoken.md) | Unsloth GGUF UD-Q4_K_M | Can't run | Can't run | 905.68 | 108.75 |
| [Strata](docs/guides/strata.md) | Unsloth GGUF UD-Q4_K_M | 1,968.80 | 105.19 | 2,000.00 | 109.13 |
| [ZML / LLMD](docs/guides/zml.md) | AMD MXFP4 | Can't run | Can't run | 774.63² | 50.16 |
| [NInfer](docs/guides/ninfer.md) | Mixed Q4/Q5/Q6/Q8 `.ninfer` | Can't run | Can't run | 556.51 | 117.08 |
| [TensorFold](docs/guides/tensorfold.md) | MLX 4-bit experts, 8-bit routers | Can't run | Can't run | 462.85 | 57.65 |

¹ The CUDA 12 figures for vLLM are from **1Cat-vLLM** running CUDA 12.8. The native Strata and llama.cpp builds used CUDA 12.9 and 12.9.2 respectively. [1Cat-vLLM measurements](docs/guides/benchmarks/vllm-graphs.json)

² LLMD prefill uses the GPU span of its complete prefill graph. The server measured 751.66 prompt tokens/s through the first token. Decode counts the 63 subsequent output tokens over the server’s decode time. [Phase timings](docs/guides/benchmarks/zml-llmd-bf16-product.json)

Compatibility applies to the listed checkpoint and upstream engine on a native V100. [Checkpoint revisions and per-request results](docs/guides/benchmarks/README.md)

[Ollama](docs/guides/ollama.md): its stock CUDA 13 runner through `vf`, measured by an outside tester (TheWeakShall) on four models, not by the protocol above. Against Ollama's stock CUDA 12 runner, prompts ran at 1.3 to 1.9 times its speed on two mixture-of-experts models and at 0.40 to 0.67 times on two dense ones, generation at 0.92 to 1.02 times.

## Validation

On V100 with driver 580.178.04, PyTorch matched the CPU on all 924 operations with a CPU reference and all 32 small models, also with `VF_CAPABILITY=8.0`. vLLM and SGLang matched the CPU reference token for token on greedy Qwen2.5-0.5B tests. Selected low-precision operations were also checked against an RTX 5090. Coverage depends on the kernel. [Validation results](docs/validation.md) · [Performance measurements](docs/performance.md) · [Known limits](LIMITS.md)

## Build and test

To build from source, install Rust 1.88 or later (edition 2024) and GCC, then run these commands from the checkout.

```sh
scripts/build.sh
scripts/fetch-all.sh redist
scripts/build-forwarders.sh
scripts/vf doctor
cargo test --workspace
```

The scripts and tests that compile CUDA sources with the toolkits of `toolkits/` go through `scripts/nvcc-host.sh`. It compiles a probe first and changes nothing when that works. On a system newer than CUDA 12.9.2, such as Ubuntu 26.04 (glibc 2.43, GCC 15), it adds what the compiler needs there, and says so: a `g++-N` that the toolkit accepts, and a header of ours that hides the C library's declarations that contradict CUDA 12's (the toolkit is not edited). Install `g++-14` there; CUDA 12.9 accepts GCC 14 at the newest. Fedora 44 and 45 have no `gcc14` package; an outside tester installed Fedora 43's, with which the GPU suite passed on Fedora 45 (TheWeakShall, [issue #4](https://github.com/nbald/volta-forever/issues/4#issuecomment-7141685965)):

```sh
sudo dnf --releasever=43 --repo=fedora --repo=updates install gcc14 gcc14-c++
```

Run `tests/gpu/run.sh` for the physical V100 suite. Use `scripts/package.sh` to create a release archive. [Test suites](tests/README.md) · [Contributing](CONTRIBUTING.md) · [Machine-code translator](docs/translator.md)

## Licence

Apache License 2.0. The instruction tables derived from denvdis are licensed under LGPL 2.1. NVIDIA and CUDA are trademarks of NVIDIA Corporation. This project is independent of NVIDIA. [License](LICENSE) · [Instruction table licence](crates/vf-tables/LICENSE) · [Notices](NOTICE)
