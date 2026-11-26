#!/usr/bin/env python3
"""VF_TRITON_TC on the host, without a GPU: how the launcher delivers python/ and what the modules
decide and do (python/sitecustomize.py, python/volta_forever_triton*.py).

  python3 tests/python/triton_tc.py [-v]

It needs the layer built (scripts/build.sh: the proxy's marker in dist/lib/libcuda.so.1, which the
module asks for) and python3 3.9 or later with venv for the checks of `vf venv`. Four groups:

  - Delivery, with an installation of the archive's layout made for the test, whose offline tool
    is a stand-in: the launcher's PYTHONPATH (set, not set, not twice, `vf env`); the import hook in
    a Python started by the launcher, in its child processes, in workers of multiprocessing (spawn,
    fork, forkserver), in a virtual environment made by `vf venv` started with no prefix; a
    sitecustomize of the user on PYTHONPATH, and one in the environment's site-packages, still run;
    nothing, and no error, under -I, -E and -S.
  - Decisions, with a stand-in for Triton (a package `triton` whose CUDABackend has the methods of
    Triton 3.3 to 3.9 and makes texts, and cubins with a table of sections): which kernels are
    compiled for sm_80 and which keep the stages of Triton; every failure of the sm_80 compilation
    leaves the kernel to Triton's stages, and what Triton's cache keeps of it; the helpers (one
    process's own children: their start, their isolation, their end, their slots, fork, the bounds
    of each wait, the program's descriptors, what a request may ask, what an answer must be, the
    temporary files); when the module does nothing.
  - Module: the rules and the checks of an answer, read in this process.
  - With a real Triton (VF_TRITON_PYTHON, a Python that has it; skipped without), through the
    launcher of this tree: kernels compiled for 7.5 (tests/python/triton_tc_compile.py): float16,
    bfloat16 and int8 products made for sm_80, with mma instructions and a cubin that keeps its
    PTX, the same cubin as an sm_80 compilation in the process; a float32 product and a kernel
    without one made for sm_75, the same cubin as with the switch off; Triton's cache, used off,
    on, off, on: the second compiles, the third and the fourth take what the first and the second
    made; a block whose pipelined kernel needs more than 96 KiB made with fewer stages;
    TRITON_DISABLE_LINE_INFO=1 keeps sm_75.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROXY = ROOT / "dist/lib/libcuda.so.1"
VOLTA = "Tesla V100-SXM2-32GB"

# The check that a Python prints: whether the hook is installed, and which sitecustomize ran.
CHECK = ("import sys; print(any(getattr(f, 'volta_forever_triton', False) for f in sys.meta_path), "
         "getattr(sys.modules.get('sitecustomize'), '__file__', None))")

STANDIN = {
    "triton/__init__.py": """
        import os as _os, sys as _sys, time as _time
        if _os.environ.get("TRITON_STANDIN_RECORD") and "--helper" in _sys.argv:
            import json as _json
            with open(_os.environ["TRITON_STANDIN_RECORD"], "w") as _f:
                _json.dump({"path": _sys.path, "isolated": _sys.flags.isolated, "no_site": _sys.flags.no_site,
                            "cwd": _os.getcwd(), "pgid": _os.getpgid(0), "sid": _os.getsid(0), "pid": _os.getpid(),
                            "ppid": _os.getppid(), "fds": sorted(int(f) for f in _os.listdir("/proc/self/fd")),
                            "env": sorted(_os.environ), "tmp": __import__("tempfile").gettempdir(),
                            "tmpmode": _os.stat(__import__("tempfile").gettempdir()).st_mode & 0o777,
                            "links": [_os.path.realpath("/proc/self/fd/" + f) for f in _os.listdir("/proc/self/fd")]}, _f)
        if _os.environ.get("TRITON_STANDIN_SLOW_IMPORT") and "--helper" in _sys.argv:
            _time.sleep(float(_os.environ["TRITON_STANDIN_SLOW_IMPORT"]))   # (a Triton slow to import, as from a cold disk)
        if _os.environ.get("TRITON_STANDIN_HANG_IMPORT") and "--helper" in _sys.argv:
            _time.sleep(3600)   # (a Triton whose import does not end, in a helper)
        if _os.environ.get("TRITON_STANDIN_FAIL_IMPORT") and "--helper" in _sys.argv:
            raise ImportError("a Triton that cannot be imported in a helper")
        __version__ = "3.99-standin"
        from .runtime import cache
        from . import backends
        """,
    "triton/runtime/__init__.py": "",
    "triton/runtime/cache.py": """
        import os

        def triton_key():
            return "3.99-sources"

        def default_cache_dir():   # (as Triton 3.3's)
            return os.path.join(os.getenv("TRITON_HOME", os.path.expanduser("~")), ".triton", "cache")
        """,
    "triton/_C/__init__.py": "",
    "triton/_C/libtriton.py": """
        class Module:
            def __init__(self, text):
                self.text = text
                self.context = None

            def __str__(self):
                return self.text


        class ir:
            @staticmethod
            def parse_mlir_module(path, context):
                with open(path) as f:
                    module = Module(f.read())
                module.context = context
                return module
        """,
    "triton/backends/__init__.py": """
        from dataclasses import dataclass
        from .nvidia import compiler


        @dataclass(frozen=True)
        class Backend:
            compiler: type
            driver: type = None


        backends = {"nvidia": Backend(compiler.CUDABackend)}
        """,
    "triton/backends/nvidia/__init__.py": "",
    "triton/backends/nvidia/compiler.py": """
        # A stand-in for Triton's NVIDIA backend: the methods of Triton 3.3 to 3.9 by their names and
        # parameters, which make texts. In the TTIR: RAISE80 makes make_ttgir fail for 8.0, SHARED=N the
        # shared memory of a stage, NOPTX a cubin without its PTX; the others are below.
        import dataclasses
        import enum
        import os
        import re
        import signal
        import struct
        import time
        from triton._C.libtriton import Module

        HOOK = None  # as Triton's knobs.runtime.add_stages_inspection_hook


        def _elf(sm, body, ptx):
            # A cubin: its ELF64 header (little-endian, CUDA's OS/ABI 0x33 and ABI 7, machine EM_CUDA
            # (190), the SM in the flags), the body, then the section .nv_debug_ptx_txt that ptxas makes
            # with -lineinfo (the PTX, its lines ended by NUL) when ptx is given, and the table of
            # sections. compile() below gives the body alone.
            names = b"\\0.shstrtab\\0" + (b".nv_debug_ptx_txt\\0" if ptx is not None else b"")
            data = b"\\0\\0" + names
            rows = [bytes(64), struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, 64 + len(body) + 2, len(names), 0, 0, 1, 0)]
            if ptx is not None:
                text = ptx.replace("\\n", "\\0").encode()
                rows.append(struct.pack("<IIQQQQIIQQ", 11, 1, 0, 0, 64 + len(body) + len(data), len(text), 0, 0, 1, 0))
                data += text
            head = (b"\\x7fELF" + bytes([2, 1, 1, 0x33, 7]) + bytes(7)
                    + struct.pack("<HHIQQQIHHHHHH", 2, 190, 1, 0, 0, 64 + len(body) + len(data), sm, 64, 0, 0, 64,
                                  len(rows), 1))
            return head + body + data + b"".join(rows)


        class Language(enum.Enum):
            TRITON = 0
            GLUON = 1


        @dataclasses.dataclass(frozen=True)
        class GPUTarget:
            backend: str
            arch: int
            warp_size: int


        @dataclasses.dataclass(frozen=True)
        class CUDAOptions:
            num_warps: int = 4
            num_ctas: int = 1
            num_stages: int = 3
            arch: str = None
            ir_override: str = None
            instrumentation_mode: str = ""
            ptx_options: str = None
            extern_libs: tuple = None
            backend_name: str = "cuda"
            debug: bool = False

            def __post_init__(self):   # (as Triton's: libdevice, of its own directory, by default)
                libs = dict(self.extern_libs or ())
                libs.setdefault("libdevice", os.path.join(os.path.dirname(__file__), "lib", "libdevice.10.bc"))
                object.__setattr__(self, "extern_libs", tuple(libs.items()))


        class CUDABackend:
            instrumentation = None

            def __init__(self, target):
                self.target = target

            def parse_options(self, opts):
                args = {"arch": "sm%d" % self.target.arch}
                args.update(opts)
                return CUDAOptions(**args)

            @staticmethod
            def make_ttgir(mod, metadata, opt, capability):
                if "NOISY80" in mod.text and capability == 80:
                    import sys as _sys
                    _sys.stderr.write("error: failures have been detected while processing an MLIR pass pipeline\\n"
                                      "note: please share the reproducer above with Triton project.\\n")
                    _sys.stderr.flush()
                    raise RuntimeError("PassManager::run failed")
                if "RAISE80" in mod.text and capability == 80:
                    raise RuntimeError("PassManager::run failed")
                if "ABORT80" in mod.text and capability == 80:
                    os.write(2, b"LLVM ERROR: Cannot select: a stand-in's\\n")
                    os.abort()  # as an LLVM or MLIR error that ends the process
                if "PTXAS80" in mod.text and capability == 80:
                    # (As Triton runs ptxas: close_fds=False, so that it holds this process's inheritable
                    # descriptors, which are recorded; then a compilation that does not end.)
                    import subprocess
                    if os.environ.get("TRITON_STANDIN_RECORD_FDS"):
                        with open(os.environ["TRITON_STANDIN_RECORD_FDS"], "w") as f:
                            for fd in os.listdir("/proc/self/fd"):
                                try:
                                    if os.get_inheritable(int(fd)):
                                        f.write(os.readlink("/proc/self/fd/" + fd) + "\\n")
                                except OSError:
                                    pass
                    subprocess.Popen(["sleep", "41.25"], close_fds=False)
                    time.sleep(3600)
                if "HANG80" in mod.text and capability == 80:
                    time.sleep(3600)
                if "DEAF80" in mod.text and capability == 80:
                    # (As LLVM's code that does not end: no handler of Python's runs meanwhile.)
                    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR2})
                    time.sleep(3600)
                if "SLEEP80" in mod.text and capability == 80:
                    time.sleep(float(os.environ.get("TRITON_STANDIN_SLEEP", "2")))
                if "LOUD80" in mod.text and capability == 80:
                    # (A compiler that wrote 512 MiB, as MLIR_ENABLE_DUMP may: a sparse file, then a crash.)
                    os.lseek(2, 512 << 20, os.SEEK_SET)
                    os.write(2, b"the end of what the compiler wrote\\n")
                    os.kill(os.getpid(), signal.SIGSEGV)
                    time.sleep(10)
                if "PTXKILLED80" in mod.text and capability == 80:
                    # (Killed from outside while "ptxas", which inherits its descriptors, runs.)
                    import subprocess
                    subprocess.Popen(["sleep", "41.25"], close_fds=False)
                    time.sleep(0.5)
                    os.kill(os.getpid(), signal.SIGKILL)
                for name in ("SIGKILL", "SIGTERM", "SIGSEGV", "SIGBUS"):
                    if name + "80" in mod.text and capability == 80:
                        os.kill(os.getpid(), getattr(signal, name))
                        time.sleep(10)
                for name, sig in (("ABORTONCE80", signal.SIGABRT), ("BUSONCE80", signal.SIGBUS)):
                    # (The first compilation of the kernel ends so, as at a lack of memory; the next does not.)
                    once = os.path.join(os.environ["HOME"], "once-" + name)
                    if name in mod.text and capability == 80 and not os.path.exists(once):
                        open(once, "w").close()
                        os.kill(os.getpid(), sig)
                        time.sleep(10)
                if "EXIT0" in mod.text and capability == 80 and not os.path.exists(os.environ["HOME"] + "/exit0-done"):
                    open(os.environ["HOME"] + "/exit0-done", "w").close()
                    os._exit(0)   # as a helper that ended on its own as the request reached it
                return Module("ttgir sm%d stages=%d\\n%s" % (capability, opt.num_stages, mod.text))

            def make_llir(self, src, metadata, options, capability):
                m = re.search(r"SHARED=(\\d+)", src.text)
                if "NOSHARED" not in src.text:
                    metadata["shared"] = (int(m.group(1)) if m else 1024) * max(1, options.num_stages)
                if "BADMD80" in src.text and capability == 80:
                    metadata["debug"] = True   # (a field of the request that no stage of Triton's sets)
                return "llir sm%d\\n%s" % (capability, src.text)

            def make_ptx(self, src, metadata, opt, capability):
                metadata["name"] = "k"
                return ".target sm_%d\\n%s" % (capability, src)

            def make_cubin(self, src, metadata, opt, capability):
                body = ("cubin sm_%d\\n" % capability).encode()
                kept = None if "NOPTX" in src else src.replace("tt.dot", "tt.dot_tampered") if "TAMPER80" in src else src
                cubin = _elf(75 if "SM75CUBIN80" in src else capability, body, kept)
                if "NOTELF80" in src and capability == 80:
                    cubin = b"not a cubin at all .nv_debug_ptx_txt" + bytes(64)
                for name, script in (("PTXKILL80", "kill -9 $$"), ("PTXSEGV80", "kill -11 $$"),
                                     ("PTXSPACE80", "echo 'ptxas fatal : Could not write output: No space left on device' >&2; exit 255")):
                    if name in src and capability == 80:
                        # As Triton's make_cubin: ptxas run, and its failure raised as an error of
                        # Triton's, with its status and what it wrote.
                        import subprocess
                        try:
                            subprocess.run(["sh", "-c", script], check=True, stderr=subprocess.PIPE)
                        except subprocess.CalledProcessError as e:
                            raise RuntimeError("`ptxas` failed with error code %d\\n`ptxas` stderr:\\n%s"
                                               % (e.returncode, e.stderr.decode()))
                if any(w in src for w in ("PTXFILE", "FIFO80", "LINK80", "SWAP80")) and capability == 80:
                    # As Triton's: the PTX in a file of the temporary directory, the cubin that ptxas
                    # writes beside it (by its path), read back with open(). FIFO80: a FIFO there in its
                    # place; LINK80: a link to another cubin; SWAP80: the directory renamed away and one
                    # of another's put in its place (with another cubin) before ptxas writes.
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".ptx") as fsrc:
                        fsrc.write(src)
                    fbin = fsrc.name + ".o"
                    if os.environ.get("TRITON_STANDIN_RECORD_TMP"):
                        with open(os.environ["TRITON_STANDIN_RECORD_TMP"], "w") as f:
                            f.write(fsrc.name)
                    forged = _elf(capability, b"cubin sm_80 forged\\n", src)
                    if "FIFO80" in src:
                        os.mkfifo(fbin)
                    elif "LINK80" in src:
                        with open(fsrc.name + ".forged", "wb") as f:
                            f.write(forged)
                        os.symlink(fsrc.name + ".forged", fbin)
                    else:
                        if "SWAP80" in src:
                            here = os.path.dirname(fsrc.name)
                            os.rename(here, here + ".renamed")
                            os.mkdir(here, 0o700)
                        time.sleep(float(os.environ.get("TRITON_STANDIN_PTXAS", "0.3")))   # (ptxas's time)
                        with open(fbin, "wb") as f:
                            f.write(forged if "SWAP80" in src else cubin)
                    with open(fbin, "rb") as f:
                        cubin = f.read()
                    os.remove(fbin)
                return cubin

            def add_stages(self, stages, options, language):
                capability = int(options.arch[2:])
                stages["ttir"] = lambda src, metadata: src
                stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, capability)
                stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, capability)
                stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.target.arch)
                stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.target.arch)
                if HOOK is not None:
                    HOOK(self, stages, options, language, capability)

            def hash(self):
                return "ptxas-%d" % self.target.arch


        def compile(ttir, arch=75, language=Language.TRITON, **options):
            # What triton.compiler.compile does with the stages, in short: (body of the cubin, metadata,
            # hash), the metadata's hash that of the key (Triton's own: of triton_key and the backend's hash).
            import hashlib
            from triton.runtime.cache import triton_key
            backend = CUDABackend(GPUTarget("cuda", arch, 32))
            backend_hash = backend.hash()
            key = "%s-%s-%s" % (triton_key(), ttir, backend_hash)
            stages = {}
            opts = backend.parse_options(options)
            backend.add_stages(stages, opts, language)
            module = Module(ttir)
            metadata = dict(dataclasses.asdict(opts), hash=hashlib.sha256(key.encode()).hexdigest())
            for name, stage in list(stages.items())[1:]:
                module = stage(module, metadata)
            if isinstance(module, bytes) and module[:4] == b"\\x7fELF":
                module = module[64:].split(b"\\0\\0", 1)[0]
            return module, metadata, backend_hash
        """,
}

# A TTIR with one tl.dot of two operands of type T.
DOT = ("tt.func public @k(%a: !tt.ptr<{t}>) {{\n"
       "  %c = tt.dot %x, %y, %acc, inputPrecision = tf32 : tensor<64x32x{t}> * tensor<32x64x{t}> -> tensor<64x64x{r}>\n"
       "  tt.return\n}}\n")


# How the sm_80 TTGIR of Triton 3.4 to 3.9 loads B from shared memory (out of a kernel compiled by Triton
# 3.8), and that of Triton 3.3.
TTGIR_B = ("#shared = #ttg.swizzled_shared<{{vec = 8, perPhase = 1, maxPhase = 8, order = [{order}]}}>\n"
           "  %b = ttg.local_load %m token %t : !ttg.memdesc<16x128xbf16, #shared, #smem, mutable, 64x128> -> "
           "tensor<16x128xbf16, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = 2}}>>\n")
TTGIR_B33 = ("#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>\n"
             "  %b = triton_gpu.local_load %m : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> "
             "tensor<64x128xbf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>\n")

# A program that compiles the TTIRs of its arguments, one after the other, and prints the targets.
COMPILE = ("import json, sys\nfrom triton.backends.nvidia import compiler as c\n"
           "print(json.dumps([c.compile(t)[0].decode()[:11] for t in sys.argv[1:]]))\n")


def dot(t, extra=""):
    return DOT.format(t=t, r="i32" if t == "i8" else "f32") + extra


def helpers(parent=None):
    """The helpers that run (pid: parent pid), of the given parent or of any."""
    out = {}
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                with open("/proc/%s/cmdline" % pid, "rb") as f:
                    cmd = f.read().split(b"\0")
                with open("/proc/%s/stat" % pid) as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            if b"--helper" in cmd and any(c.endswith(b"volta_forever_triton_helper.py") for c in cmd):
                if parent is None or ppid == parent:
                    out[int(pid)] = ppid
    return out


def zombies_of(pid):
    n = 0
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                with open("/proc/%s/stat" % p) as f:
                    st = f.read().rsplit(")", 1)[1].split()
            except OSError:
                continue
            n += st[0] == "Z" and int(st[1]) == pid
    return n


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PROXY.is_file():
            raise unittest.SkipTest("the layer is not built (scripts/build.sh)")
        cls.scratch = tempfile.TemporaryDirectory(prefix="vf-triton-tc-test-")
        cls.root = Path(cls.scratch.name)
        # An installation of the archive's layout: the launcher, python/, the proxy (for its marker)
        # and a stand-in for the offline tool, which prints the file tool-says of the installation.
        home = cls.root / "vf-home"
        (home / "bin").mkdir(parents=True)
        (home / "lib").mkdir()
        shutil.copy2(ROOT / "scripts/vf", home / "bin/vf")
        shutil.copytree(ROOT / "python", home / "python", ignore=shutil.ignore_patterns("__pycache__"))
        (home / "python/__pycache__").write_text("")  # as scripts/package.sh makes it
        (home / "lib/libcuda.so.1").symlink_to(PROXY)
        (home / "VERSION").write_text("0.0.1-test\n")
        tool = home / "bin/vf-translate"
        tool.write_text('#!/bin/sh\ncat "$(dirname "$0")/../tool-says"\n')
        tool.chmod(0o755)
        cls.says("k.cubin: PTX for sm_80 retargeted to sm_70 (1 modules, 16 tensor-core instructions rewritten, "
                 "0 tanh expanded, 0 functions made to trap)\n", home)
        cls.home = home
        cls.vf = home / "bin/vf"
        standin = cls.root / "standin"
        for name, text in STANDIN.items():
            (standin / name).parent.mkdir(parents=True, exist_ok=True)
            (standin / name).write_text(textwrap.dedent(text))
        cls.standin = standin
        cls.cache = cls.root / "triton-cache"
        (cls.root / "tmp").mkdir()

    @classmethod
    def tearDownClass(cls):
        mine = helpers()
        for pid in mine:   # (none should be left: a test that fails may leave one; only ours are ended)
            try:
                with open("/proc/%d/environ" % pid, "rb") as f:
                    if str(cls.root).encode() in f.read():
                        os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
        cls.scratch.cleanup()

    @staticmethod
    def says(text, home):
        (home / "tool-says").write_text(text)

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("VF_", "TRITON_", "VOLTA_FOREVER", "PYTHON"))}
        env.update(VF_QUIET="1", VF_TRITON_TC="1", VF_TRITON_TC_GPUS=VOLTA, PYTHONDONTWRITEBYTECODE="1",
                   HOME=str(self.root), TMPDIR=str(self.root / "tmp"), TRITON_CACHE_DIR=str(self.cache))
        env.update(extra)
        return {k: v for k, v in env.items() if v is not None}

    def run_vf(self, args, env=None, **kw):
        return subprocess.run([str(self.vf)] + args, env=env or self.env(), capture_output=True, text=True,
                              timeout=kw.pop("timeout", 120), **kw)

    def own(self, cache=None):
        """The directory of this user's helpers in a Triton cache."""
        return Path(cache or self.cache) / ("volta-forever-%d" % os.geteuid())


class Delivery(Base):
    def expected(self, bare, ours):
        """The file of the sitecustomize module after start: the one Python runs without the
        layer's (`bare`), or, on a Python that has none (the uv and PyPy builds), the layer's own
        (`ours`), which stays in sys.modules: Python imports sitecustomize once, at start, and the
        import system wants the module it ran there afterwards."""
        return ours if bare == "None" else bare

    def check(self, out):
        """(hook installed, file of the sitecustomize module) from the line of CHECK."""
        hook, _, sc = out.strip().splitlines()[-1].partition(" ")
        return hook == "True", sc

    def test_pythonpath(self):
        r = self.run_vf(["sh", "-c", 'echo "[$PYTHONPATH]"'], self.env(PYTHONPATH="/a:/b"))
        self.assertEqual(r.stdout.strip(), "[%s/python:/a:/b]" % self.home)
        r = self.run_vf(["sh", "-c", 'echo "[${PYTHONPATH-unset}]"'], self.env(VF_TRITON_TC="0"))
        self.assertEqual(r.stdout.strip(), "[unset]")
        r = self.run_vf(["sh", "-c", 'echo "[${PYTHONPATH-unset}]"'], self.env(VF_TRITON_TC=None))
        self.assertEqual(r.stdout.strip(), "[unset]")
        # Under itself: not twice.
        r = self.run_vf([str(self.vf), "sh", "-c", 'echo "[$PYTHONPATH]"'])
        self.assertEqual(r.stdout.strip(), "[%s/python]" % self.home)
        r = self.run_vf(["env"])
        self.assertIn("export PYTHONPATH='%s/python'" % self.home, r.stdout)
        r = self.run_vf(["env"], self.env(VF_TRITON_TC="0"))
        self.assertNotIn("PYTHONPATH", r.stdout)

    def test_hook_in_python_and_children(self):
        r = self.run_vf([sys.executable, "-c", CHECK])
        self.assertEqual(self.check(r.stdout)[0], True, r.stdout + r.stderr)
        r = self.run_vf([sys.executable, "-c", CHECK], self.env(VF_TRITON_TC="0"))
        self.assertEqual(self.check(r.stdout)[0], False, r.stdout + r.stderr)
        child = "import subprocess, sys; print(subprocess.run([sys.executable, '-c', %r], capture_output=True, " \
                "text=True).stdout)" % CHECK
        r = self.run_vf([sys.executable, "-c", child])
        self.assertEqual(self.check(r.stdout)[0], True, r.stdout + r.stderr)

    def test_multiprocessing(self):
        script = self.root / "workers.py"
        script.write_text(textwrap.dedent("""
            import multiprocessing, sys
            def hooked(_):
                return any(getattr(f, "volta_forever_triton", False) for f in sys.meta_path)
            if __name__ == "__main__":
                for method in ("spawn", "fork", "forkserver"):
                    with multiprocessing.get_context(method).Pool(2) as pool:
                        print(method, all(pool.map(hooked, range(2))))
            """))
        r = self.run_vf([sys.executable, str(script)])
        self.assertEqual(r.stdout.split(), ["spawn", "True", "fork", "True", "forkserver", "True"], r.stderr)

    def test_flags_that_skip_site(self):
        for flag in ("-I", "-E", "-S"):
            r = self.run_vf([sys.executable, flag, "-c", "import sys; print(any(getattr(f, 'volta_forever_triton', "
                             "False) for f in sys.meta_path))"])
            self.assertEqual((r.stdout.strip(), r.returncode), ("False", 0), flag)
            self.assertNotIn("volta-forever", r.stderr, flag)

    def test_what_a_start_costs(self):
        # The modules that a Python started with the switch imports beyond those it imports
        # anyway: the layer's sitecustomize alone; nothing of Triton's patch until Triton is
        # imported (a Python without a sitecustomize of its own imported threading and re).
        code = "import sys; print(' '.join(sorted(sys.modules)))"
        e = {k: v for k, v in self.env().items() if k != "PYTHONDONTWRITEBYTECODE"}
        bare = set(subprocess.run([sys.executable, "-c", code], env=dict(e, VF_TRITON_TC="0"),
                                  capture_output=True, text=True).stdout.split())
        r = self.run_vf([sys.executable, "-c", code], e)
        extra = set(r.stdout.split()) - bare - {"sitecustomize"}
        self.assertEqual(extra, set(), r.stderr)
        # Triton imported: volta_forever_triton is read, not the client of the helpers, which a
        # process reads at its first kernel that may be compiled for sm_80, nor the helper's module,
        # which no program reads (Python compiles each from its source: what a process does not read
        # costs it nothing). Nothing is written into the installation: python/ keeps its five
        # files, the stand-in of Triton imported and bytecode written where it may be.
        code = textwrap.dedent("""
            import sys
            import triton
            from triton.backends.nvidia import compiler as c
            v = sys.modules.get("volta_forever_triton")
            print(v is not None, v is not None and "client_module" in v._state)
            c.compile(sys.argv[1])
            m = v._state.get("client_module")
            print(getattr(m, "__file__", None), sorted(k for k in sys.modules if "volta" in k))
            """)
        r = self.run_vf([sys.executable, "-c", code, dot("f16")], dict(e, PYTHONPATH=str(self.standin)))
        self.assertEqual(r.stdout.strip().splitlines(), [
            "True False", "%s ['volta_forever_triton']" % (self.home / "python/volta_forever_triton_client.py")], r.stderr)
        self.assertEqual(sorted(os.listdir(self.home / "python")),
                         ["__pycache__", "sitecustomize.py", "volta_forever_triton.py", "volta_forever_triton_client.py",
                          "volta_forever_triton_helper.py"])
        self.assertTrue((self.home / "python/__pycache__").is_file())

    def test_two_copies_of_python(self):
        # Two copies of the layer's python/ on PYTHONPATH (two installations, a vf venv of another
        # release, eval "$(vf env)" of one and vf of another): one start, no recursion, the
        # distribution's sitecustomize run once.
        other = self.root / "other-python"
        if not other.exists():
            shutil.copytree(ROOT / "python", other, ignore=shutil.ignore_patterns("__pycache__"))
        bare = subprocess.run([sys.executable, "-c", "import sys, importlib.util; s = importlib.util.find_spec("
                               "'sitecustomize'); print(s.origin if s else None)"], env=self.env(VF_TRITON_TC=None),
                              capture_output=True, text=True).stdout.strip()
        code = CHECK + "; print(sys.modules.get('sitecustomize') is not None)"
        e = self.env(PYTHONPATH="%s:%s" % (other, self.home / "python"))
        r = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=60)
        self.assertNotIn("RecursionError", r.stderr)
        self.assertNotIn("Error in sitecustomize", r.stderr)
        self.assertEqual(self.check(r.stdout.splitlines()[0]), (True, self.expected(bare, str(other / "sitecustomize.py"))),
                         r.stderr)
        # Through the launcher: the other copy is taken out of PYTHONPATH, this one put first.
        r = self.run_vf([sys.executable, "-c", "import os; print(os.environ['PYTHONPATH'])"], e)
        self.assertEqual(r.stdout.strip(), "%s/python" % self.home, r.stderr)
        r = self.run_vf([sys.executable, "-c", "import os; print(os.environ.get('PYTHONPATH'))"],
                        self.env(VF_TRITON_TC="0", PYTHONPATH="/a:%s:/b" % other))
        self.assertEqual(r.stdout.strip(), "/a:/b", r.stderr)

    def test_users_sitecustomize_still_runs(self):
        user = self.root / "user-site"
        user.mkdir(exist_ok=True)
        (user / "sitecustomize.py").write_text("import os\nos.environ['USER_SITECUSTOMIZE'] = 'ran'\nNAME = 'user'\n")
        code = CHECK + "; import os, sitecustomize; print(os.environ.get('USER_SITECUSTOMIZE'), sitecustomize.NAME)"
        r = self.run_vf([sys.executable, "-c", code], self.env(PYTHONPATH=str(user)))
        lines = r.stdout.strip().splitlines()
        self.assertEqual(lines[-1], "ran user", r.stdout + r.stderr)
        self.assertEqual(self.check(lines[0]), (True, str(user / "sitecustomize.py")))
        # One that fails: as Python says of it without the layer's, and the hook is there all the same.
        (user / "sitecustomize.py").write_text("raise ValueError('of the user')\n")
        r = self.run_vf([sys.executable, "-c", CHECK], self.env(PYTHONPATH=str(user)))
        self.assertIn("Error in sitecustomize", r.stderr)
        self.assertIn("ValueError: of the user", r.stderr)
        self.assertEqual(self.check(r.stdout)[0], True)

    def test_a_finders_sitecustomize_still_runs(self):
        # A sitecustomize that no directory of sys.path holds, given by a finder that a .pth
        # installed after Python's own (an editable install's): run as Python would run it, which
        # is when the directories of sys.path have none (a Python without a distribution's own).
        try:
            import venv  # noqa: F401
        except ImportError:
            self.skipTest("no venv module")
        ve = self.root / "ve-finder"
        if not (ve / "bin/python").exists():
            r = subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(ve)], capture_output=True, text=True)
            if not (ve / "bin/python").exists():
                self.skipTest("python3 -m venv failed: " + r.stderr[-300:])
        site = Path(subprocess.run([str(ve / "bin/python"), "-c", "import sysconfig; print(sysconfig.get_path("
                                    "'purelib'))"], capture_output=True, text=True).stdout.strip())
        hidden = self.root / "hidden-site"
        hidden.mkdir(exist_ok=True)
        (hidden / "sitecustomize.py").write_text("NAME = 'finder'\n")
        (site / "_vf_test_finder.py").write_text(textwrap.dedent("""
            import importlib.machinery, sys
            class Finder:
                @classmethod
                def find_spec(cls, name, path=None, target=None):
                    if name != "sitecustomize":
                        return None
                    return importlib.machinery.PathFinder.find_spec(name, [%r])
            sys.meta_path.append(Finder)
            """ % str(hidden)))
        (site / "_vf_test_finder.pth").write_text("import _vf_test_finder\n")
        code = "import sys; m = sys.modules.get('sitecustomize'); print(getattr(m, 'NAME', None), getattr(m, '__file__', None))"
        bare = subprocess.run([str(ve / "bin/python"), "-c", code], env=self.env(VF_TRITON_TC=None),
                              capture_output=True, text=True).stdout.strip()
        e = self.env(PYTHONPATH=str(self.home / "python"))
        r = subprocess.run([str(ve / "bin/python"), "-c", code], env=e, capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), bare, r.stderr)

    def test_reload_of_the_users_sitecustomize(self):
        # importlib.reload of the user's sitecustomize runs the user's code again in the same module,
        # and leaves nothing of the layer's in it.
        user = self.root / "user-site-reload"
        user.mkdir(exist_ok=True)
        (user / "sitecustomize.py").write_text("import os\nCOUNT = int(os.environ.get('VF_TEST_COUNT', '0')) + 1\n"
                                               "os.environ['VF_TEST_COUNT'] = str(COUNT)\n")
        code = textwrap.dedent("""
            import importlib, sys, sitecustomize
            m = sitecustomize
            r = importlib.reload(m)
            print(r is m, m.COUNT, m.__file__, sorted(k for k in vars(m) if 'volta' in k.lower()))
            """)
        r = subprocess.run([sys.executable, "-c", code], env=self.env(PYTHONPATH="%s:%s" % (self.home / "python", user)),
                           capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "True 2 %s []" % (user / "sitecustomize.py"), r.stderr)

    def test_the_systems_sitecustomize_still_runs(self):
        bare = subprocess.run([sys.executable, "-c", "import sys, importlib.util; s = importlib.util.find_spec("
                               "'sitecustomize'); print(s.origin if s else None)"], env=self.env(VF_TRITON_TC=None),
                              capture_output=True, text=True).stdout.strip()
        r = self.run_vf([sys.executable, "-c", CHECK])
        self.assertEqual(self.check(r.stdout), (True, self.expected(bare, str(self.home / "python/sitecustomize.py"))))
        if bare == "None":
            # Where Python has none, `import sitecustomize` finds the layer's: a module without a
            # public name (the only difference a program can see, besides that the import works).
            r = self.run_vf([sys.executable, "-c", "import sitecustomize; print([k for k in vars(sitecustomize) "
                             "if not k.startswith('_')])"])
            self.assertEqual(r.stdout.strip(), "[]", r.stderr)

    def test_venv(self):
        try:
            import venv  # noqa: F401
        except ImportError:
            self.skipTest("no venv module")
        ve = self.root / "ve"
        r = self.run_vf(["venv", str(ve), "--", "--without-pip"])
        if not (ve / "bin/python").exists():
            self.skipTest("python3 -m venv failed: " + r.stderr[-300:])
        # The sitecustomize that this Python runs without the layer's (a distribution's, in the
        # directory of the standard library, comes before the environment's site-packages).
        bare = subprocess.run([str(ve / "bin/python"), "-c", "import importlib.util; s = importlib.util.find_spec("
                               "'sitecustomize'); print(s.origin if s else None)"],
                              env=self.env(VF_VENV="0", VF_TRITON_TC=None), capture_output=True, text=True).stdout.strip()
        # Started with no prefix: the environment's hook starts it again through the launcher.
        r = subprocess.run([str(ve / "bin/python"), "-c", CHECK + "; import os; print(os.environ.get('VOLTA_FOREVER'))"],
                           env=self.env(), capture_output=True, text=True)
        lines = r.stdout.strip().splitlines()
        self.assertEqual(lines[-1], "0.0.1-test", r.stdout + r.stderr)
        self.assertEqual(self.check(lines[0]), (True, self.expected(bare, str(self.home / "python/sitecustomize.py"))))
        r = subprocess.run([str(ve / "bin/python"), "-c", CHECK], env=self.env(VF_TRITON_TC="0"), capture_output=True,
                           text=True)
        self.assertEqual(self.check(r.stdout)[0], False, r.stdout + r.stderr)



class Decisions(Base):
    def compile(self, ttir, env=None, **options):
        """Compile a TTIR with the stand-in, through the launcher: (cubin, metadata, hash, lines of the layer)."""
        log = self.root / "vf.log"
        if log.exists():
            log.unlink()
        code = textwrap.dedent("""
            import json, os, sys
            from triton.backends.nvidia import compiler as c
            from triton.runtime.cache import triton_key
            if os.environ.get("STANDIN_HOOK"):
                def hook(backend, stages, options, language, capability):
                    llir = stages["llir"]
                    stages["llir"] = lambda src, metadata: llir(src, metadata)
                c.HOOK = hook
            cubin, md, h = c.compile(sys.argv[1], **json.loads(sys.argv[2]))
            print(json.dumps([cubin.decode(), md, h + "|" + triton_key(),
                              c.CUDABackend.__dict__.get("_volta_forever", False)]))
            """)
        e = self.env(PYTHONPATH=str(self.standin), VF_LOG=str(log), VF_LOG_LEVEL="3")
        e.update(env or {})
        r = self.run_vf([sys.executable, "-c", code, ttir, json.dumps(options)], e)
        self.assertEqual(r.returncode, 0, r.stderr)
        cubin, md, h, patched = json.loads(r.stdout.strip().splitlines()[-1])
        lines = log.read_text() if log.exists() else ""
        return cubin, md, h, patched, lines

    def assertTarget(self, result, sm):
        self.assertTrue(result[0].startswith("cubin sm_%d" % sm), result)

    def test_types(self):
        for t in ("f16", "bf16", "i8"):
            r = self.compile(dot(t))
            self.assertTarget(r, 80)
            self.assertIn("Triton kernel k: compiled for sm_80", r[4])
            self.assertRegex(r[2], r"^ptxas-75-vf-tc3-f16\+bf16\+i8-[^|]+\|3\.99-sources-vf-tc3-f16\+bf16\+i8-")
        for t in ("f32", "f64"):
            r = self.compile(dot(t))
            self.assertTarget(r, 75)
            self.assertIn("a tl.dot of %s by %s" % (t, t), r[4])
            self.assertRegex(r[2], r"^ptxas-75-vf-tc3-")  # the mark is on every kernel told 7.5
        r = self.compile(dot("f16"), env=dict(VF_TRITON_TC="bf16,i8"))
        self.assertTarget(r, 75)
        self.assertRegex(r[2], r"^ptxas-75-vf-tc3-bf16\+i8-")
        r = self.compile(dot("bf16"), env=dict(VF_TRITON_TC="bf16,i8,f32x"))
        self.assertTarget(r, 80)
        self.assertIn("f32x is not one of", r[4])
        mixed = dot("f16").replace("tt.return", "%d = tt.dot %x, %y, %acc : tensor<64x32xf32> * tensor<32x64xf32> -> "
                                   "tensor<64x64xf32>\n  tt.return")
        self.assertTarget(self.compile(mixed), 75)
        self.assertTarget(self.compile("tt.func public @k() {\n  tt.return\n}\n"), 75)
        self.assertTarget(self.compile(dot("f16", "%f = tt.fp_to_fp %x : tensor<64xf8E4M3FN>\n")), 75)
        self.assertTarget(self.compile(dot("f16", "%s = tt.dot_scaled %x\n")), 75)

    def test_what_triton_is_given(self):
        # Told 8.0: as Triton makes it, and no mark.
        r = self.compile(dot("f16"), arch=80)
        self.assertTarget(r, 80)
        self.assertRegex(r[2], r"^ptxas-80\|3\.99-sources-vf-tc3-")
        # Instrumentation, an IR given for a stage, the settings that read or replace what a stage
        # makes: Triton's own stages.
        self.assertTarget(self.compile(dot("f16"), instrumentation_mode="consan"), 75)
        self.assertTarget(self.compile(dot("f16"), ir_override="/x.ttgir"), 75)
        for name, value in (("TRITON_KERNEL_OVERRIDE", "1"), ("TRITON_DISABLE_LINE_INFO", "1"), ("USE_IR_LOC", "ttgir")):
            self.assertTarget(self.compile(dot("f16"), env={name: value}), 75)

    def test_override_arch_is_kept(self):
        code = textwrap.dedent("""
            import json
            from triton.backends.nvidia import compiler as c
            b = c.CUDABackend(c.GPUTarget("cuda", 75, 32))
            o = c.CUDAOptions(arch="sm80")
            stages = {}
            b.add_stages(stages, o, c.Language.TRITON)
            print(all(f.__qualname__.endswith("add_stages.<locals>.<lambda>") for f in stages.values()))
            """)
        r = self.run_vf([sys.executable, "-c", code], self.env(PYTHONPATH=str(self.standin)))
        self.assertEqual(r.stdout.strip(), "True", r.stderr)

    def test_gluon_is_kept(self):
        code = textwrap.dedent("""
            from triton.backends.nvidia import compiler as c
            print(c.compile(%r, language=c.Language.GLUON)[0])
            """ % dot("f16"))
        r = self.run_vf([sys.executable, "-c", code], self.env(PYTHONPATH=str(self.standin)))
        self.assertTrue(r.stdout.strip().startswith("b'cubin sm_75"), r.stdout + r.stderr)

    def test_stages_for_shared_memory(self):
        r = self.compile(dot("f16", "SHARED=40000\n"), num_stages=4)
        self.assertTarget(r, 80)
        self.assertEqual((r[1]["shared"], r[1]["num_stages"]), (80000, 2))
        self.assertIn("compiled for sm_80, 2 stages of the 4 asked, for its shared memory", r[4])
        r = self.compile(dot("f16", "SHARED=99000\n"), num_stages=3)
        self.assertTarget(r, 75)
        self.assertEqual(r[1]["shared"], 297000)  # Triton's own stages: what Triton says of it, as without the layer
        self.assertIn("99000 bytes of shared memory with 1 stages", r[4])

    def test_failures_leave_the_kernel_to_triton(self):
        r = self.compile(dot("f16", "RAISE80\n"))
        self.assertTarget(r, 75)
        self.assertEqual(r[4].count("compiled for sm_75: an error of its sm_80 compilation: RuntimeError: PassManager::run "
                                    "failed"), 1, r[4])
        self.assertEqual(r[1]["shared"], 3 * 1024)  # the metadata of Triton's stages alone
        r = self.compile(dot("f16", "NOPTX\n"))
        self.assertTarget(r, 75)
        self.assertIn("its cubin does not keep its PTX", r[4])
        try:
            self.says("k.cubin: PTX for sm_80 retargeted to sm_70 (1 modules, 0 tensor-core instructions rewritten, "
                      "0 tanh expanded, 0 functions made to trap)\n  refused kernel: k (k: an instruction)\n", self.home)
            r = self.compile(dot("f16"))
            self.assertTarget(r, 75)
            self.assertIn("refused kernel: k", r[4])
            self.says("k.cubin: PTX for sm_80 retargeted to sm_70 (1 modules, 0 tensor-core instructions rewritten, "
                      "0 tanh expanded, 1 functions made to trap)\n  trap: k: an instruction\n", self.home)
            self.assertTarget(self.compile(dot("f16")), 75)
            self.says("k.cubin: machine code for sm_80 translated to sm_70 (1 modules, 1 functions, 0 refused, "
                      "0 with instructions added)\n", self.home)
            self.assertTarget(self.compile(dot("f16")), 75)
        finally:
            self.says("k.cubin: PTX for sm_80 retargeted to sm_70 (1 modules, 16 tensor-core instructions rewritten, "
                      "0 tanh expanded, 0 functions made to trap)\n", self.home)

    def test_when_nothing_is_done(self):
        r = self.compile(dot("f16"), env=dict(VF_TRITON_TC_GPUS=VOLTA + ",Tesla T4"))
        self.assertTarget(r, 75)
        self.assertEqual(r[2], "ptxas-75|3.99-sources")
        self.assertIn("a GPU that is not a Volta (Tesla T4)", r[4])
        r = self.compile(dot("f16"), env=dict(VF_CAPABILITY="8.0"))
        self.assertEqual((r[0][:10], r[2]), ("cubin sm_7", "ptxas-75|3.99-sources"))
        self.assertIn("VF_CAPABILITY is 8.0", r[4])
        r = self.compile(dot("f16"), env=dict(VF_TRITON_TC_GPUS=""))
        self.assertEqual(r[2], "ptxas-75|3.99-sources")
        # Without the layer (PYTHONPATH by hand, NVIDIA's libcuda.so.1 or none): nothing.
        code = "from triton.backends.nvidia import compiler as c; print(c.compile(%r)[0], c.compile(%r)[2])" % (
            dot("f16"), dot("f16"))  # (the backend's hash: no mark)
        e = self.env(PYTHONPATH="%s:%s" % (self.home / "python", self.standin))
        r = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True)
        self.assertEqual(r.stdout.split()[-1], "ptxas-75", r.stdout + r.stderr)
        self.assertIn("is not the layer's", r.stderr)

    def test_guards(self):
        # A stage that a hook put in place of Triton's (Triton's add_stages_inspection_hook): the
        # stages are left as they are, and the layer says so once.
        r = self.compile(dot("f16"), env={"STANDIN_HOOK": "1"})
        self.assertTarget(r, 75)
        self.assertIn("are not the ones this layer knows", r[4])
        # More than one CTA (Hopper's clusters): left as it is.
        self.assertTarget(self.compile(dot("f16"), num_ctas=2), 75)
        # A compilation that does not say how much shared memory its kernel needs: Triton's stages.
        r = self.compile(dot("f16", "NOSHARED\n"))
        self.assertTarget(r, 75)
        self.assertIn("its shared memory is not known", r[4])

    def test_vf_safe(self):
        # VF_SAFE=1 (exactly), given to Python behind the launcher's back: no hook (sitecustomize);
        # set by the program before it imports Triton: the hook is there, the module does nothing
        # and says why. Another value: nothing changes.
        code = textwrap.dedent("""
            import json, os, sys
            if sys.argv[2] != "-":
                os.environ["VF_SAFE"] = sys.argv[2]
            from triton.backends.nvidia import compiler as c
            print(json.dumps([c.compile(sys.argv[1])[0].decode()[:11], c.CUDABackend.__dict__.get("_volta_forever", False)]))
            """)
        path = "PYTHONPATH=%s" % os.pathsep.join([str(self.home / "python"), str(self.standin)])
        seen = []
        for given, inside in (("1", "-"), ("-", "1"), ("yes", "-"), (" 1", "-"), ("-", "1 ")):
            args = ["--", "env", path] + (["VF_SAFE=" + given] if given != "-" else []) + [sys.executable, "-c", code,
                                                                                   dot("f16"), inside]
            r = self.run_vf(args)
            self.assertEqual(r.returncode, 0, r.stderr)
            seen.append(tuple(json.loads(r.stdout.strip().splitlines()[-1])))
            if inside == "1":
                self.assertIn("VF_TRITON_TC is not used: VF_SAFE=1", r.stderr)
        self.assertEqual(seen, [("cubin sm_75", False), ("cubin sm_75", True), ("cubin sm_80", True),
                                ("cubin sm_80", True), ("cubin sm_80", True)])

    def test_slower_products_stay_at_75(self):
        # The kinds of products of SLOWER, measured slower for sm_80 on a V100: a bfloat16 product of a
        # tile of 16 x 128 x 64 whose B is stored by rows, and a float16 one with float16 sums of 16 x
        # 128, are compiled for 7.5, and the line says why; the same with B stored by columns, or with
        # float32 sums, are compiled for sm_80.
        def kernel(t, acc, order):
            return ("tt.func public @k(%%a: !tt.ptr<%s>) {\n  %%c = tt.dot %%x, %%y, %%acc : tensor<16x64x%s> * "
                    "tensor<64x128x%s> -> tensor<16x128x%s>\n  tt.return\n}\n" % (t, t, t, acc)) + TTGIR_B.format(order=order)
        r = self.compile(kernel("bf16", "f32", "1, 0"))
        self.assertTarget(r, 75)
        self.assertIn("compiled for sm_75: bfloat16 products of a skinny tile measured slower", r[4])
        self.assertTarget(self.compile(kernel("bf16", "f32", "0, 1")), 80)
        # The same tile with 2 stages, measured faster: sm_80 (the rule of version 2 left it to 7.5).
        self.assertTarget(self.compile(kernel("bf16", "f32", "1, 0"), num_stages=2), 80)
        r = self.compile(kernel("f16", "f16", "1, 0"))
        self.assertTarget(r, 75)
        self.assertIn("compiled for sm_75: float16 products with float16 sums", r[4])
        self.assertTarget(self.compile(kernel("f16", "f32", "1, 0")), 80)

    def test_a_crash_of_the_sm80_compiler(self):
        # The sm_80 compilation ends its process (an LLVM or MLIR error: abort): it runs in a
        # helper process, the program goes on with Triton's own compilation of the kernel, and says
        # once why. The same for one that does not end in VF_TRITON_TC_TIMEOUT seconds.
        r = self.compile(dot("f16", "ABORT80\n"))
        self.assertTarget(r, 75)
        self.assertIn("compiled for sm_75: its sm_80 compilation ended its process", r[4])
        r = self.compile(dot("f16", "HANG80\n"), env={"VF_TRITON_TC_TIMEOUT": "3"})
        self.assertTarget(r, 75)
        self.assertIn("its sm_80 compilation did not end in 3 s", r[4])
        # The next kernel of the same process gets a helper of its own.
        code = textwrap.dedent("""
            import json, sys
            from triton.backends.nvidia import compiler as c
            a = c.compile(sys.argv[1])[0].decode()
            b = c.compile(sys.argv[2])[0].decode()
            print(json.dumps([a[:11], b[:11]]))
            """)
        e = self.env(PYTHONPATH=str(self.standin))
        r = self.run_vf([sys.executable, "-c", code, dot("f16", "ABORT80\n"), dot("f16")], e)
        self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1]), ["cubin sm_75", "cubin sm_80"], r.stderr)



    def test_a_fork_while_another_thread_holds_the_lock(self):
        # A fork() in one thread while another holds the module's lock (it does for a moment in each
        # line said, each decision): the child gets the lock free, and compiles.
        code = textwrap.dedent("""
            import json, os, sys, threading, time
            from triton.backends.nvidia import compiler as c
            v = sys.modules["volta_forever_triton"]
            c.compile(sys.argv[1])
            def hold():
                with v._lock:
                    time.sleep(2)
            threading.Thread(target=hold).start()
            time.sleep(0.3)
            pid = os.fork()
            if pid == 0:
                os._exit(0 if c.compile(sys.argv[2])[0].decode()[:11] == "cubin sm_80" else 3)
            end = time.monotonic() + 10
            while time.monotonic() < end:
                done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    break
                time.sleep(0.05)
            else:
                os.kill(pid, 9)
                os.waitpid(pid, 0)
                status = "hung"
            print(json.dumps(status))
            """)
        r = self.run_vf([sys.executable, "-W", "ignore", "-c", code, dot("f16"), dot("bf16")],
                        self.env(PYTHONPATH=str(self.standin)))
        self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1]), 0, r.stderr)


    def test_what_the_compiler_writes(self):
        # What Triton's compiler writes when its sm_80 compilation fails (an MLIR dump, "please share
        # the reproducer") does not reach the program's standard error at the default level: one line
        # says what came of the kernel; with VF_LOG_LEVEL=3 the text is written too.
        code = "import sys\nfrom triton.backends.nvidia import compiler as c\nprint(c.compile(sys.argv[1])[0].decode()[:11])\n"
        for level in ("1", "3"):
            r = self.run_vf([sys.executable, "-c", code, dot("f16", "NOISY80\n")],
                            self.env(PYTHONPATH=str(self.standin),
                                     VF_LOG_LEVEL=level))
            self.assertEqual(r.stdout.strip(), "cubin sm_75", r.stderr)
            self.assertIn("compiled for sm_75: an error of its sm_80 compilation: RuntimeError: PassManager::run failed "
                          "(VF_LOG_LEVEL=3 shows what the compiler wrote)", r.stderr)
            self.assertEqual("please share the reproducer" in r.stderr, level == "3", r.stderr)

    def test_much_written_before_a_crash(self):
        # A helper that wrote 512 MiB before it crashed (MLIR_ENABLE_DUMP, which the helper is given,
        # writes every pass's IR): its waiter reads the last 20,000 bytes of it alone, from its end,
        # for the line of VF_LOG_LEVEL=3.
        code = textwrap.dedent("""
            import json, resource, sys
            from triton.backends.nvidia import compiler as c
            before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            out = c.compile(sys.argv[1])[0].decode()[:11]
            print(json.dumps([out, (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) // 1024]))
            """)
        r = self.run_vf([sys.executable, "-c", code, dot("f16", "LOUD80\n")],
                        self.env(PYTHONPATH=str(self.standin), VF_LOG_LEVEL="3"))
        out, grown = self.last(r)
        self.assertEqual(out, "cubin sm_75", r.stderr[-2000:])
        self.assertLess(grown, 200, "MB")
        self.assertIn("the end of what the compiler wrote", r.stderr)
        self.assertIn("compiled for sm_75: its sm_80 compilation ended its process (SIGSEGV)", r.stderr)

    def test_the_helpers_triton_is_the_programs(self):
        # A program whose sys.path, after it imported Triton, names another Triton first: its helper
        # imports the program's all the same (from the directory that holds it), and compiles.
        other = self.root / "other-triton"
        if not other.exists():
            shutil.copytree(self.standin, other)
            init = other / "triton/__init__.py"
            init.write_text(init.read_text().replace('"3.99-standin"', '"3.98-other"'))
        code = textwrap.dedent("""
            import json, sys
            from triton.backends.nvidia import compiler as c
            sys.path.insert(0, sys.argv[2])
            print(json.dumps([c.compile(sys.argv[1])[0].decode()[:11]] * 1))
            """)
        r = self.run_vf([sys.executable, "-c", code, dot("f16"), str(other)],
                        self.env(PYTHONPATH=str(self.standin)))
        self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1]), ["cubin sm_80"], r.stderr)
        self.assertNotIn("another Triton", r.stderr)

    def test_a_triton_changed_under_the_program(self):
        # The files of the program's Triton changed on disk while it runs, with the same version
        # (another build reinstalled: its NVIDIA backend's module replaced; an editable install
        # edited: its library written in place): the next helper's Triton is not taken (a stat of
        # each, recorded when the program imported Triton), the kernel is compiled by Triton's own
        # stages, and the process compiles as without the switch, said once.
        code = textwrap.dedent("""
            import json, os, sys, time
            from triton.backends.nvidia import compiler as c
            first = c.compile(sys.argv[1])[0].decode()[:24]
            if sys.argv[3] == "backend":
                path = c.__file__
                text = open(path).read().replace('body = ("cubin sm_%d\\\\n" % capability).encode()',
                                                 'body = ("cubin sm_%d OTHER BUILD\\\\n" % capability).encode()')
                assert "OTHER BUILD" in text
                open(path + ".new", "w").write(text)
                os.replace(path + ".new", path)
            else:
                with open(sys.modules["triton._C.libtriton"].__file__, "a") as f:
                    f.write("# edited\\n")
            for h in sys.modules["volta_forever_triton"].client()._the_pool().helpers:
                h.kill()   # (as at its idle time: the next kernel starts another)
            time.sleep(0.5)
            print(json.dumps([first, c.compile(sys.argv[2])[0].decode()[:24], c.compile(sys.argv[2])[2]]))
            """)
        for what, name in (("backend", "its NVIDIA backend"), ("library", "its library")):
            other = self.root / ("standin-changed-" + what)
            if other.exists():
                shutil.rmtree(other)
            shutil.copytree(self.standin, other, ignore=shutil.ignore_patterns("__pycache__"))
            r = self.run_vf([sys.executable, "-c", code, dot("f16"), dot("bf16"), what], self.env(PYTHONPATH=str(other)))
            first, second, key = self.last(r)
            self.assertEqual((first, second, key), ("cubin sm_80\n", "cubin sm_75\n", "ptxas-75"), (what, r.stderr))
            self.assertEqual(r.stderr.count("every kernel of this process is compiled by Triton's own stages: the "
                                            "program's Triton changed on disk since the program imported it (%s"
                                            % name), 1, r.stderr)

    def test_the_helper_and_the_programs_start(self):
        # A helper runs nothing of the program's start (-I -S): a user's sitecustomize that writes to
        # standard output runs in the program alone, and the helper's answers are not mixed with what
        # the program writes.
        site = self.root / "users-site"
        site.mkdir(exist_ok=True)
        (site / "sitecustomize.py").write_text("print('users-site', flush=True)\n")
        code = "import sys\nfrom triton.backends.nvidia import compiler as c\nprint(c.compile(sys.argv[1])[0].decode()[:11])\n"
        e = self.env(PYTHONPATH=os.pathsep.join([str(self.standin), str(site)]))
        r = self.run_vf([sys.executable, "-c", code, dot("f16")], e)
        self.assertEqual(r.stdout.split(), ["users-site", "cubin", "sm_80"], r.stderr)


    def test_many_descriptors(self):
        # A program with 1,100 descriptors open (more than select() takes): the kernel gets sm_80, no
        # traceback, and no descriptor is left open by the compilation.
        code = textwrap.dedent("""
            import json, os, resource, sys
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 4096), hard), hard))
            fds = [os.open(os.devnull, os.O_RDONLY) for _ in range(1100)]
            from triton.backends.nvidia import compiler as c
            def fds():
                out = {}
                for f in os.listdir("/proc/self/fd"):
                    try:
                        out[f] = os.readlink("/proc/self/fd/" + f)
                    except OSError:
                        pass
                return out
            c.compile(sys.argv[1])   # (what a first compilation imports, which may keep a file)
            before = fds()
            out = [c.compile(sys.argv[2])[0].decode()[:11], c.compile(sys.argv[3])[0].decode()[:11]]
            after = fds()
            print(json.dumps([out, sorted(v for k, v in after.items() if k not in before)]))
            """)
        r = self.run_vf([sys.executable, "-c", code, dot("i8"), dot("f16"), dot("bf16")],
                        self.env(PYTHONPATH=str(self.standin)))
        self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1]), [["cubin sm_80"] * 2, []], r.stderr)
        self.assertNotIn("Traceback", r.stderr)


    def test_a_ptxas_ended_from_outside(self):
        # ptxas killed from outside (SIGKILL, as by the out-of-memory killer, which picks the largest
        # process) or failing on a full disk is not an error of the kernel: compiled for 7.5 this
        # time, not kept in Triton's cache. ptxas that crashes (SIGSEGV) is: kept.
        for name, kept, why in (("PTXKILL80", False, "ptxas was ended from outside (signal 9)"),
                                ("PTXSPACE80", False, "no space left on its device"),
                                ("PTXSEGV80", True, "an error of its sm_80 compilation: RuntimeError: `ptxas` failed with "
                                                    "error code -11")):
            r = self.compile(dot("f16", name + "\n"))
            self.assertTarget(r, 75)
            self.assertIn(why, r[4])
            self.assertEqual("not kept in Triton's cache" in r[4], not kept, r[4])

    def test_a_compilation_that_does_not_end(self):
        # VF_TRITON_TC_TIMEOUT bounds the wait: the helper that compiles is killed with what it runs
        # (ptxas, which Triton runs with close_fds=False): its process group. What ptxas inherits holds
        # nothing of the helper's (its socket, its locks): standard input, output and error alone.
        t = time.monotonic()
        fds = self.root / "ptxas-fds"
        r = self.compile(dot("f16", "PTXAS80\n"), env={"VF_TRITON_TC_TIMEOUT": "3", "TRITON_STANDIN_RECORD_FDS": str(fds)})
        inherited = fds.read_text().split()
        self.assertFalse([x for x in inherited if "socket:" in x or "slot-" in x or "waiting-" in x], inherited)
        took = time.monotonic() - t
        self.assertTarget(r, 75)
        self.assertIn("its sm_80 compilation did not end in 3 s", r[4])
        self.assertLess(took, 12)
        self.assertEqual(self.ptxas_left(), [])

    def ptxas_left(self, within=5.0):
        """The stand-in's ptxas (sleep 41.25) of this test's helpers that still run, once a kill of
        their group has had the time to end them (SIGKILL ends a process when it is next scheduled,
        which a loaded machine may delay)."""
        end = time.monotonic() + within
        while True:
            left = [p for p in os.listdir("/proc") if p.isdigit() and self.cmdline(p)[:2] == [b"sleep", b"41.25"]
                    and str(self.root).encode() in self.environ(int(p))]
            if not left or time.monotonic() > end:
                return left
            time.sleep(0.05)

    def test_a_helper_killed_while_its_ptxas_runs(self):
        # A helper killed while ptxas runs (which Triton starts with close_fds=False): ptxas holds none
        # of its descriptors, so that the program sees its end at once (not at the time-out), and what
        # its group still runs is killed with it.
        t = time.monotonic()
        r = self.compile(dot("f16", "PTXKILLED80\n"), env={"VF_TRITON_TC_TIMEOUT": "20"})
        self.assertTarget(r, 75)
        self.assertIn("not kept in Triton's cache: its sm_80 compilation was ended from outside (signal 9)", r[4])
        self.assertLess(time.monotonic() - t, 10)
        self.assertEqual(self.ptxas_left(), [])

    def test_how_a_compilation_ended(self):
        # A crash of the compiler (SIGSEGV, SIGBUS, as SIGABRT above) is the kernel's: Triton's own
        # stages, kept in Triton's cache. A compilation ended from outside (SIGKILL as from the
        # out-of-memory killer, SIGTERM as to a group of processes) is not: Triton's own stages for
        # this compilation, not kept.
        for name, kept in (("SIGSEGV", True), ("SIGBUS", True), ("SIGKILL", False), ("SIGTERM", False)):
            r = self.compile(dot("f16", name + "80\n"))
            self.assertTarget(r, 75)
            if kept:
                self.assertEqual(r[4].count("compiled for sm_75: its sm_80 compilation ended its process (%s)" % name), 1,
                                 r[4])   # (said once, at the default level)
                self.assertNotIn("not kept", r[4])
            else:
                number = {"SIGKILL": 9, "SIGTERM": 15}[name]
                self.assertIn("compiled for sm_75, not kept in Triton's cache: its sm_80 compilation was ended "
                              "from outside (signal %d)" % number, r[4])

    def test_an_abort_is_tried_again_once(self):
        # A helper ended by SIGABRT or SIGBUS, which a lack of resources gives too (LLVM aborts when
        # an allocation fails; a mapped library truncated under the helper gives SIGBUS), is not the
        # kernel's at once: the kernel is compiled again in a new helper, once (said at
        # VF_LOG_LEVEL=2), and gets sm_80 when that ends well; a second such end is the kernel's
        # (Triton's own stages, kept). SIGSEGV is the kernel's at once.
        for name, sig in (("ABORTONCE80", "SIGABRT"), ("BUSONCE80", "SIGBUS")):
            once = self.root / ("once-" + name)
            if once.exists():
                once.unlink()
            r = self.compile(dot("f16", name + "\n"))
            self.assertTarget(r, 80)
            self.assertEqual(r[4].count("its sm_80 compilation ended its process (%s), which a lack of memory may give "
                                        "too: compiled again in a new helper" % sig), 1, r[4])
            self.assertNotIn("not kept", r[4])
        for name, sig, again in (("ABORT80", "SIGABRT", 1), ("SIGBUS80", "SIGBUS", 1), ("SIGSEGV80", "SIGSEGV", 0)):
            r = self.compile(dot("f16", name + "\n"))
            self.assertTarget(r, 75)
            self.assertEqual(r[4].count("compiled again in a new helper"), again, r[4])
            self.assertIn("compiled for sm_75: its sm_80 compilation ended its process (%s)" % sig, r[4])
            self.assertNotIn("not kept", r[4])


    def test_options_that_a_helper_does_not_take(self):
        # A kernel whose options a helper would refuse (ptxas's that name a file, libraries out of
        # Triton's directory) is compiled by Triton's own stages, said once; one whose ptxas options
        # are of those Triton gives, for sm_80.
        r = self.compile(dot("f16"), ptx_options="--options-file /etc/hostname")
        self.assertTarget(r, 75)
        self.assertIn("a kernel is compiled for sm_75 when its ptx_options give ptxas '--options-file'", r[4])
        r = self.compile(dot("f16"), extern_libs=(("libdevice", "/etc/hostname"),))
        self.assertTarget(r, 75)
        self.assertIn("extern_libs name /etc/hostname, which is not a file of Triton's directory", r[4])
        self.assertTarget(self.compile(dot("f16"), ptx_options="-v -lineinfo --maxrregcount=128"), 80)


    def test_a_fifo_where_the_cubin_is(self):
        # A FIFO (or anything else than a regular file) where ptxas writes the cubin is not read as
        # one: the compilation fails at once (it would wait for a writer until its time-out), and the
        # kernel goes to Triton's own stages, not kept.
        t = time.time()
        r = self.compile(dot("f16", "FIFO80\n"), env={"VF_TRITON_TC_TIMEOUT": "20"})
        self.assertTarget(r, 75)
        self.assertIn("is not a regular file", r[4])
        self.assertIn("not kept in Triton's cache", r[4])
        self.assertLess(time.time() - t, 15)

    def test_the_layers_settings(self):
        # A setting that keeps the layer from running the PTX that the tool checked: nothing done.
        for name, value in (("VF_TRANSLATE", "first"), ("VF_TRANSLATE", "force"), ("VF_ADAPT", "0")):
            r = self.compile(dot("f16"), env={name: value})
            self.assertEqual((r[0][:11], r[2]), ("cubin sm_75", "ptxas-75|3.99-sources"), name)
            self.assertIn("%s=%s" % (name, value), r[4])
        # A setting that changes what the layer makes of the PTX, or what its tool says of it: a mark
        # of its own, so that a cache made with another value is not taken.
        marks = set()
        for extra in ({}, {"VF_PAIRS": "0"}, {"VF_ALIGNED_ROWS": "0"}, {"VF_BF16_TC_GROUPS": "1"},
                      {"VF_TRANSLATE": "0"}, {"VF_LAZY": "0"}):
            r = self.compile(dot("f16"), env=extra)
            self.assertTrue(r[0].startswith("cubin sm_80"), extra)
            marks.add(r[2])
        self.assertEqual(len(marks), 6, marks)

    def test_a_triton_that_differs(self):
        other = self.root / "standin-other"
        if not other.exists():
            shutil.copytree(self.standin, other)
            p = other / "triton/backends/nvidia/compiler.py"
            p.write_text(p.read_text().replace("def make_ptx(self, src, metadata, opt, capability):",
                                               "def make_ptx(self, src, metadata, opt, capability, extra=None):"))
        code = "from triton.backends.nvidia import compiler as c; from triton.runtime.cache import triton_key; " \
               "print(triton_key(), c.compile(%r)[0], c.CUDABackend.__dict__.get('_volta_forever', False))" % dot("f16")
        r = self.run_vf([sys.executable, "-c", code], self.env(PYTHONPATH=str(other)))
        self.assertTrue(r.stdout.strip().startswith("3.99-sources b'cubin sm_75"), r.stdout + r.stderr)
        self.assertTrue(r.stdout.strip().endswith("False"))
        self.assertIn("is not one this layer knows (CUDABackend.make_ptx takes self, src, metadata, opt, capability, "
                      "extra): VF_TRITON_TC is not used", r.stderr)
        self.assertEqual(r.stderr.count("is not one this layer knows"), 1)



    # --- the helpers -------------------------------------------------------------------------

    def script(self, name, text):
        path = self.root / (name + ".py")
        path.write_text(textwrap.dedent(text))
        return str(path)

    def last(self, r):
        self.assertTrue(r.stdout.strip(), r.stderr)
        return json.loads(r.stdout.strip().splitlines()[-1])

    def gone(self, pids, within=10.0):
        """Whether none of these processes runs (nor is a zombie) within so many seconds."""
        end = time.monotonic() + within
        while time.monotonic() < end:
            if not any(os.path.exists("/proc/%d" % p) for p in pids):
                return True
            time.sleep(0.05)
        return False

    def test_helpers_of_a_process_tree(self):
        # Two threads compiling at once, a child made by fork() once the program has helpers, two
        # processes that the program starts, workers of multiprocessing (spawn, forkserver, fork) and
        # the program after them: each gets the sm_80 kernel from helpers of its own (children of that
        # process); the child made by fork() keeps no copy of its parent's helpers' sockets; none of
        # the program's children is left a zombie, and its helpers end with it.
        program = self.script("tree", """
            import json, multiprocessing, os, subprocess, sys, threading
            from triton.backends.nvidia import compiler as c
            TTIR = sys.argv[1]
            def helpers(parent):
                out = []
                for p in os.listdir("/proc"):
                    if p.isdigit():
                        try:
                            cmd = open("/proc/%s/cmdline" % p, "rb").read()
                            ppid = int(open("/proc/%s/stat" % p).read().rsplit(")", 1)[1].split()[1])
                        except (OSError, ValueError, IndexError):
                            continue
                        if b"--helper" in cmd and ppid == parent:
                            out.append(int(p))
                return out
            def one(_):
                return c.compile(TTIR)[0].decode()[:11] + (" own" if helpers(os.getpid()) else " none")
            def sockets():
                out = set()
                for f in os.listdir("/proc/self/fd"):
                    try:
                        out.add(os.readlink("/proc/self/fd/" + f))
                    except OSError:
                        pass
                return out
            if __name__ == "__main__":
                out = []
                ts = [threading.Thread(target=lambda: out.append(c.compile(TTIR)[0].decode()[:11])) for _ in range(2)]
                [t.start() for t in ts]; [t.join() for t in ts]
                vft = sys.modules["volta_forever_triton"].client()
                theirs = {"socket:[%d]" % h.inode[1] for h in vft._the_pool().helpers}
                mine = helpers(os.getpid())
                r, w = os.pipe()
                pid = os.fork()
                if pid == 0:
                    kept = sorted(theirs & sockets())
                    got = one(0)
                    os.write(w, json.dumps([got, kept]).encode())
                    os._exit(0)
                os.close(w)
                _, status = os.waitpid(pid, 0)
                child = json.loads(os.read(r, 65536).decode())
                procs = [subprocess.Popen([sys.executable, "-c", "import sys\\nfrom triton.backends.nvidia import compiler "
                                           "as c\\nprint(c.compile(sys.argv[1])[0].decode()[:11])", TTIR],
                                          stdout=subprocess.PIPE, text=True) for _ in range(2)]
                out += [p.communicate()[0].strip() for p in procs]
                for method in ("spawn", "forkserver", "fork"):
                    with multiprocessing.get_context(method).Pool(2) as pool:
                        out += pool.map(one, range(2))
                out.append(c.compile(TTIR)[0].decode()[:11])
                zombies = 0
                for p in os.listdir("/proc"):
                    if p.isdigit():
                        try:
                            st = open("/proc/%s/stat" % p).read().rsplit(")", 1)[1].split()
                        except OSError:
                            continue
                        zombies += st[0] == "Z" and int(st[1]) == os.getpid()
                print(json.dumps([out, child, status, zombies, len(mine), helpers(os.getpid())]))
            """)
        r = self.run_vf([sys.executable, program, dot("f16")], self.env(PYTHONPATH=str(self.standin)))
        out, child, status, zombies, mine, left = self.last(r)
        self.assertEqual(out, ["cubin sm_80"] * 4 + ["cubin sm_80 own"] * 6 + ["cubin sm_80"], r.stderr)
        self.assertEqual((child, status, zombies), (["cubin sm_80 own", []], 0, 0), r.stderr)
        self.assertIn(mine, (1, 2))
        self.assertTrue(self.gone(left), left)

    def test_a_helper_started_by_a_thread_that_ended(self):
        # PR_SET_PDEATHSIG is sent when the thread that started the helper ends: the helper goes on
        # serving the process (it ends only when the process does), and the next kernel uses it.
        code = textwrap.dedent("""
            import json, os, sys, threading, time
            from triton.backends.nvidia import compiler as c
            out = []
            t = threading.Thread(target=lambda: out.append(c.compile(sys.argv[1])[0].decode()[:11]))
            t.start(); t.join()
            vft = sys.modules["volta_forever_triton"].client()
            first = [h.pid for h in vft._the_pool().helpers]
            time.sleep(1)
            out.append(c.compile(sys.argv[2])[0].decode()[:11])
            print(json.dumps([out, first == [h.pid for h in vft._the_pool().helpers], len(first)]))
            """)
        r = self.run_vf([sys.executable, "-c", code, dot("f16"), dot("bf16")], self.env(PYTHONPATH=str(self.standin)))
        self.assertEqual(self.last(r), [["cubin sm_80"] * 2, True, 1], r.stderr)

    def test_a_helper_ends_with_its_program(self):
        # A program killed (SIGKILL) while its helper compiles, and its ptxas runs: both end with it
        # (PR_SET_PDEATHSIG, then the helper's whole group). A program that ends while a thread of it
        # waits for a compilation: nothing on its standard error, it ends at once, and so does its
        # helper.
        code = textwrap.dedent("""
            import os, sys, threading, time
            from triton.backends.nvidia import compiler as c
            threading.Thread(target=lambda: c.compile(sys.argv[1]), daemon=True).start()
            vft = None
            while not vft or not vft._the_pool().helpers:
                time.sleep(0.05)
                vft = getattr(sys.modules["volta_forever_triton"], "_state", {}).get("client_module")
            time.sleep(1)
            print(vft._the_pool().helpers[0].pid, flush=True)
            time.sleep(float(sys.argv[2]))
            print("bye")
            """)
        e = self.env(PYTHONPATH=str(self.standin), VF_TRITON_TC_TIMEOUT="60")
        p = subprocess.Popen([str(self.vf), sys.executable, "-c", code, dot("f16", "PTXAS80\n"), "60"], env=e,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            helper = int(p.stdout.readline())
            sleeps = [int(x) for x in os.listdir("/proc") if x.isdigit() and self.cmdline(x)[:2] == [b"sleep", b"41.25"]
                      and self.pgid(x) == helper]
            self.assertEqual(len(sleeps), 1)
            p.kill()
            p.wait()
        finally:
            p.kill()
            p.wait()
            p.stdout.close()
            p.stderr.close()
        self.assertTrue(self.gone([helper] + sleeps), (helper, sleeps))
        r = self.run_vf([sys.executable, "-c", code, dot("f16", "HANG80\n"), "0"], e)
        lines = r.stdout.strip().splitlines()
        self.assertEqual(lines[-1], "bye", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertNotIn("Error", r.stderr.replace("NVIDIA's libcuda.so not found", ""))
        self.assertTrue(self.gone([int(lines[0])]))

    def test_a_helper_that_cannot_see_its_program_end(self):
        # A helper whose compilation does not end, in code that runs no handler of Python's (its
        # PR_SET_PDEATHSIG is then not handled), while its program is killed: it ends by itself after
        # twice VF_TRITON_TC_TIMEOUT (SIGALRM, whatever it runs), and leaves its place.
        code = textwrap.dedent("""
            import sys, threading, time
            from triton.backends.nvidia import compiler as c
            threading.Thread(target=lambda: c.compile(sys.argv[1]), daemon=True).start()
            vft = None
            while not vft or not vft._the_pool().helpers or not vft._the_pool().helpers[0].ready:
                time.sleep(0.05)
                vft = sys.modules["volta_forever_triton"]._state.get("client_module")
            time.sleep(0.5)
            print(vft._the_pool().helpers[0].pid, flush=True)
            time.sleep(60)
            """)
        p = subprocess.Popen([str(self.vf), sys.executable, "-c", code, dot("f16", "DEAF80\n")],
                             env=self.env(PYTHONPATH=str(self.standin), VF_TRITON_TC_TIMEOUT="3"), stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True)
        try:
            helper = int(p.stdout.readline())
            p.kill()
            p.wait()
            t = time.monotonic()
            self.assertFalse(self.gone([helper], 2))
            self.assertTrue(self.gone([helper], 10))
            self.assertLess(time.monotonic() - t, 10)
        finally:
            p.kill()
            p.wait()
            p.stdout.close()

    def cmdline(self, pid):
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
                return f.read().split(b"\0")
        except OSError:
            return []

    def pgid(self, pid):
        try:
            with open("/proc/%s/stat" % pid) as f:
                return int(f.read().rsplit(")", 1)[1].split()[2])
        except (OSError, ValueError, IndexError):
            return None

    def test_the_helper_ends_when_idle(self):
        # A helper with nothing to do for VF_TRITON_TC_IDLE seconds ends, and is waited for (no
        # zombie); the next kernel starts another. One killed while it is idle is replaced at the next
        # kernel. One that ends on its own as a request reaches it (its idle time up) is replaced, and
        # the kernel still gets sm_80.
        code = textwrap.dedent("""
            import json, os, signal, sys, time
            from triton.backends.nvidia import compiler as c
            vft = None
            def helper():
                return [h.pid for h in sys.modules["volta_forever_triton"].client()._the_pool().helpers]
            out = [c.compile(sys.argv[1])[0].decode()[:11]]
            first = helper()
            time.sleep(3.5)
            zombies = 0
            for p in os.listdir("/proc"):
                if p.isdigit():
                    try:
                        st = open("/proc/%s/stat" % p).read().rsplit(")", 1)[1].split()
                    except OSError:
                        continue
                    zombies += st[0] == "Z" and int(st[1]) == os.getpid()
            gone = not os.path.exists("/proc/%d" % first[0]) and helper() == []
            out.append(c.compile(sys.argv[2])[0].decode()[:11])
            second = helper()
            os.kill(second[0], signal.SIGKILL)
            time.sleep(0.5)
            out.append(c.compile(sys.argv[3])[0].decode()[:11])
            third = helper()
            out.append(c.compile(sys.argv[4])[0].decode()[:11])
            print(json.dumps([out, gone, zombies, len(set(first + second + third)), len(helper())]))
            """)
        r = self.run_vf([sys.executable, "-c", code, dot("f16"), dot("bf16"), dot("i8"), dot("f16", "EXIT0\n")],
                        self.env(PYTHONPATH=str(self.standin), VF_TRITON_TC_IDLE="2"))
        self.assertEqual(self.last(r), [["cubin sm_80"] * 4, True, 0, 3, 1], r.stderr)
        self.assertNotIn("not kept", r.stderr)

    def test_a_helper_that_cannot_start(self):
        # A helper whose Triton cannot be imported: Triton's own stages for every kernel of the
        # process, said once with why, with no traceback, in each process. The process then compiles
        # as without the switch: the key of every kernel after the first has no mark, and only the
        # first, whose key had it, is not kept in Triton's cache.
        code = textwrap.dedent("""
            import json, sys
            from triton.backends.nvidia import compiler as c
            out = [c.compile(t) for t in sys.argv[1:]]
            v = sys.modules["volta_forever_triton"]
            print(json.dumps([[o[0].decode()[:11] for o in out], [o[2] for o in out],
                              len(v._state.get("not_kept", ())) // 2]))
            """)
        e = self.env(PYTHONPATH=str(self.standin), TRITON_STANDIN_FAIL_IMPORT="1")
        for _ in range(2):
            r = self.run_vf([sys.executable, "-c", code, dot("f16"), dot("bf16"), dot("i8")], e)
            out, keys, not_kept = self.last(r)
            self.assertEqual(out, ["cubin sm_75"] * 3, r.stderr)
            self.assertEqual(r.stderr.count("every kernel of this process is compiled by Triton's own stages: its helper "
                                            "did not start: ImportError: a Triton that cannot be imported"), 1, r.stderr)
            self.assertEqual(len([x for x in r.stderr.splitlines() if x.startswith("[volta-forever] Triton kernel")]), 0,
                             r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertIn("-vf-tc", keys[0])
            self.assertEqual(keys[1:], ["ptxas-75"] * 2)
            self.assertEqual(not_kept, 1)

    def test_an_executable_that_is_not_the_interpreter(self):
        # A frozen application (sys.frozen), or one whose sys.executable is not the interpreter that
        # runs it (an application, or a script named python3), is not started again: the switch is
        # not used in that process, said once, decided before any kernel's key has the mark (Triton's
        # cache keeps its kernels as without the switch), and the application was not run.
        app = self.root / "myapp"
        app.write_text("#!/bin/sh\ntouch %s/myapp-ran\nsleep 60\n" % self.root)
        app.chmod(0o755)
        fake = self.root / "fake" / "python3"
        fake.parent.mkdir(exist_ok=True)
        fake.write_text("#!/bin/sh\ntouch %s/myapp-ran\nexec sleep 60\n" % self.root)
        fake.chmod(0o755)
        code = textwrap.dedent("""
            import json, sys, time
            how = sys.argv[2]
            if how == "frozen":
                sys.frozen = True
            else:
                sys.executable = how
            from triton.backends.nvidia import compiler as c
            t = time.time()
            out = [c.compile(sys.argv[1]), c.compile(sys.argv[1])]
            print(json.dumps([[o[0].decode()[:11] for o in out], [o[2] for o in out], time.time() - t]))
            """)
        for how, why in (("frozen", "this program is a frozen application"),
                         (str(app), "is not known to be the Python interpreter of this process"),
                         (str(fake), "is not known to be the Python interpreter of this process")):
            r = self.run_vf([sys.executable, "-c", code, dot("f16"), how], self.env(PYTHONPATH=str(self.standin)))
            out, keys, seconds = self.last(r)
            self.assertEqual((out, keys), (["cubin sm_75"] * 2, ["ptxas-75"] * 2), r.stderr)
            self.assertEqual(r.stderr.count("VF_TRITON_TC is not used: "), 1, r.stderr)
            self.assertEqual(r.stderr.count(why), 1, r.stderr)
            self.assertLess(seconds, 5)
        self.assertFalse((self.root / "myapp-ran").exists())

    def test_a_helper_slow_to_import_triton(self):
        # A process waits for its first helper's import of Triton VF_TRITON_TC_START seconds at most
        # (that kernel, and the next ones until the helper is ready, compiled by Triton's own stages,
        # not kept, said once), and gets sm_80 once it is ready, from that helper.
        code = textwrap.dedent("""
            import json, sys, time
            from triton.backends.nvidia import compiler as c
            out, took = [], []
            for t, pause in ((sys.argv[1], 0), (sys.argv[2], 5), (sys.argv[3], 0)):
                t0 = time.monotonic()
                out.append(c.compile(t)[0].decode()[:11])
                took.append(round(time.monotonic() - t0, 1))
                time.sleep(pause)
            pool = sys.modules["volta_forever_triton"].client()._the_pool()
            print(json.dumps([out, took, len(pool.helpers)]))
            """)
        e = self.env(PYTHONPATH=str(self.standin), TRITON_STANDIN_SLOW_IMPORT="4", VF_TRITON_TC_START="1.5")
        r = self.run_vf([sys.executable, "-c", code, dot("f16"), dot("bf16"), dot("i8")], e)
        out, took, n = self.last(r)
        self.assertEqual((out, n), (["cubin sm_75", "cubin sm_75", "cubin sm_80"], 1), r.stderr)
        self.assertTrue(1.4 <= took[0] < 3 and took[1] < 0.5, took)
        self.assertEqual(r.stderr.count("a helper still imports Triton after 1.5 s"), 1, r.stderr)
        self.assertEqual(r.stderr.count("compiled for sm_75, not kept"), 0, r.stderr)   # (each at level 2)

    def test_an_import_that_does_not_end(self):
        # A helper whose import of Triton does not end (a network disk that hangs) ends after
        # VF_TRITON_TC_TIMEOUT seconds by itself (SIGALRM): the process then compiles as without the
        # switch, said once; no helper is left.
        code = textwrap.dedent("""
            import json, sys, time
            from triton.backends.nvidia import compiler as c
            a = c.compile(sys.argv[1])[0].decode()[:11]
            pids = [h.pid for h in sys.modules["volta_forever_triton"].client()._the_pool().helpers]
            time.sleep(4)
            b = c.compile(sys.argv[2])
            print(json.dumps([a, b[0].decode()[:11], b[2], pids]))
            """)
        e = self.env(PYTHONPATH=str(self.standin), TRITON_STANDIN_HANG_IMPORT="1", VF_TRITON_TC_START="1",
                     VF_TRITON_TC_TIMEOUT="2")
        r = self.run_vf([sys.executable, "-c", code, dot("f16"), dot("bf16")], e)
        a, b, key, pids = self.last(r)
        self.assertEqual((a, b, key), ("cubin sm_75", "cubin sm_75", "ptxas-75"), r.stderr)
        self.assertEqual(r.stderr.count("its helper ended at its start (status -14)"), 1, r.stderr)
        self.assertTrue(self.gone(pids))

    def test_what_the_helper_runs_of_the_program(self):
        # A helper runs nothing of the program's start (-I -S: no PYTHON* setting, no user site, no
        # sitecustomize, no .pth file), gets nothing of its environment but the settings that a
        # compilation reads (no secret, no LD_PRELOAD or LD_AUDIT) and nothing of its sys.path but the
        # directory that holds the program's Triton, after the standard library; its working
        # directory is /, it has its own session and process group, and of the program's
        # descriptors only its socket (and the locks of its slot and of "waiting"): not a file that
        # the program left inheritable. Its temporary directory is its own (mode 0700), in TMPDIR.
        users = self.root / "users-path"
        users.mkdir(exist_ok=True)
        record = self.root / "helper-saw.json"
        code = textwrap.dedent("""
            import json, os, sys
            fd = os.open(sys.argv[2], os.O_RDONLY)
            os.set_inheritable(fd, True)
            from triton.backends.nvidia import compiler as c
            print(json.dumps([c.compile(sys.argv[1])[0].decode()[:11], os.getpid()]))
            """)
        r = self.run_vf(["--", "env", "LD_PRELOAD=libm.so.6", "FOO_SECRET=1", "PYTHONWARNINGS=ignore", sys.executable,
                         "-c", code, dot("f16"), str(self.root / "vf-home/VERSION")],
                        self.env(PYTHONPATH="%s:%s" % (self.standin, users), TRITON_STANDIN_RECORD=str(record)))
        target, program = self.last(r)
        self.assertEqual(target, "cubin sm_80", r.stderr)
        saw = json.loads(record.read_text())
        self.assertEqual((saw["isolated"], saw["no_site"], saw["cwd"], saw["ppid"]), (1, 1, "/", program))
        self.assertEqual((saw["pgid"], saw["sid"]), (saw["pid"], saw["pid"]))
        self.assertEqual(saw["path"][-1], str(self.standin))
        self.assertNotIn(str(users), saw["path"])
        self.assertTrue(any(os.path.isfile(os.path.join(p, "os.py")) for p in saw["path"][:-1]), saw["path"])
        self.assertEqual([k for k in ("LD_PRELOAD", "LD_AUDIT", "PYTHONPATH", "PYTHONWARNINGS", "FOO_SECRET",
                                      "TRITON_CACHE_DIR") if k in saw["env"]], [])
        # /dev/null; its socket, slot and lock "waiting" (in the directory of the user's helpers); its
        # directory and the file of what it writes (in TMPDIR); and the stand-in's record, the
        # listing itself, and what the interpreter keeps open itself (/dev/urandom for CPython 3.9,
        # its own file for 3.14 and PyPy): nothing of the program's.
        python = (os.path.realpath(sys.executable), os.path.dirname(os.path.realpath(sys.executable)) + "/")
        tmp = str(self.root / "tmp") + "/"
        for x in saw["links"]:
            self.assertTrue(x in ("/dev/null", "/dev/urandom", str(record)) or x.startswith((str(self.own()) + "/", tmp))
                            or x.startswith("/proc/") or x == python[0] or x.startswith(python[1]), (x, saw["links"]))
        self.assertEqual(sorted(os.path.basename(x).split("-")[0] for x in saw["links"] if x.startswith(str(self.own()))),
                         ["slot", "waiting"], saw["links"])
        self.assertNotIn(str(self.root / "vf-home/VERSION"), saw["links"])
        self.assertEqual(len([x for x in saw["links"] if "socket:" in x]), 1, saw["links"])
        self.assertTrue(saw["tmp"].startswith(tmp), saw["tmp"])
        self.assertEqual(saw["tmpmode"], 0o700)

    def test_the_helper_writes_no_bytecode_where_the_program_writes_none(self):
        # A program that writes no bytecode (PYTHONDONTWRITEBYTECODE, -B): its helper, which -I would
        # not tell, writes none into the program's Triton either (a directory whose contents change
        # makes the program's import system read it again); one that writes it under a prefix
        # (PYTHONPYCACHEPREFIX, given relative to the program's working directory): its helper, whose
        # working directory is /, writes it under that prefix too; one that writes it beside the
        # sources: as Python does.
        prefix = self.root / "bytecode-prefix"
        for dont, given, want in (("1", None, False), (None, os.path.relpath(prefix), False), (None, None, True)):
            standin = self.root / ("standin-bytecode-%s-%s" % (dont, given is not None))
            if standin.exists():
                shutil.rmtree(standin)
            shutil.copytree(self.standin, standin, ignore=shutil.ignore_patterns("__pycache__"))
            r = self.run_vf([sys.executable, "-c", COMPILE, dot("f16")],
                            self.env(PYTHONPATH=str(standin), PYTHONDONTWRITEBYTECODE=dont, PYTHONPYCACHEPREFIX=given))
            self.assertEqual(self.last(r), ["cubin sm_80"], r.stderr)
            self.assertEqual(any(standin.rglob("__pycache__")), want, (dont, given))
            if given is not None:   # (the program wrote its own there: the prefix was in force)
                self.assertTrue(any(prefix.rglob("*.pyc")), given)

    def test_the_temporary_files(self):
        # Triton's PTX, and the cubin that ptxas writes, are in the helper's own directory (mode 0700)
        # in the temporary directory (TMPDIR), where Triton's own compilation writes them, removed
        # when the helper ends. The directory of the user's helpers in Triton's cache directory
        # holds the lock files of the places alone: nothing else is made there, and nothing that is
        # there is removed (a directory named as a helper's of this host whose pid is not alive
        # here, as one of another PID namespace or of another host of the same name would be, is
        # left as it is). The same when that directory cannot be written.
        import socket
        host = __import__("re").sub(r"[^A-Za-z0-9_.]", "_", socket.gethostname())[:64]
        dead = subprocess.run(["sh", "-c", "echo $$"], capture_output=True, text=True).stdout.strip()
        own = self.own()
        own.mkdir(mode=0o700, parents=True, exist_ok=True)
        (own / ("%s-%s-left" % (host, dead))).mkdir(exist_ok=True)
        seen = self.root / "ptx-path"
        e = self.env(PYTHONPATH=str(self.standin), TRITON_STANDIN_RECORD_TMP=str(seen))
        for extra, mode in (("", 0o700), (" // 2", 0o500)):
            os.chmod(own, mode)
            try:
                r = self.run_vf([sys.executable, "-c", COMPILE, dot("f16", "PTXFILE%s\n" % extra)], e)
            finally:
                os.chmod(own, 0o700)
            self.assertEqual(self.last(r), ["cubin sm_80"], r.stderr)
            work = os.path.dirname(seen.read_text())
            self.assertEqual(os.path.dirname(work), str(self.root / "tmp"), work)
            self.assertTrue(self.gone_path(work), work)
            self.assertEqual(sorted(p.name for p in own.iterdir() if not p.name.startswith(("slot-", "waiting-"))),
                             ["%s-%s-left" % (host, dead)])
        # The temporary directory as Python's tempfile finds it in the program, where Triton's own
        # compilation writes: TMP when TMPDIR is not set, or what the program set tempfile.tempdir to.
        other = self.root / "tmp-other"
        other.mkdir(exist_ok=True)
        code = "import sys, tempfile\ngiven = sys.argv.pop()\nif given != '-':\n    tempfile.tempdir = given\n" + COMPILE
        for unset, given in ((True, "-"), (False, str(other))):
            r = self.run_vf([sys.executable, "-c", code, dot("f16", "PTXFILE // 3\n"), given],
                            dict({k: v for k, v in e.items() if k != "TMPDIR"}, TMP=str(other)) if unset else e)
            self.assertEqual(self.last(r), ["cubin sm_80"], r.stderr)
            self.assertEqual(os.path.dirname(os.path.dirname(seen.read_text())), str(other), seen.read_text())

    def gone_path(self, path, within=5.0):
        end = time.monotonic() + within
        while os.path.exists(path) and time.monotonic() < end:
            time.sleep(0.05)
        return not os.path.exists(path)

    def test_a_cache_directory_that_cannot_be_used(self):
        # Triton's cache directory where the directory of the user's helpers cannot be made (a file),
        # or that directory a link (not followed): the places are not counted in that process (said
        # once, at VF_LOG_LEVEL=2), and its kernels are compiled for sm_80 by helpers without one.
        afile = self.root / "a-file"
        afile.write_text("")
        linked = self.root / "cache-with-link"
        linked.mkdir(exist_ok=True)
        if not (linked / ("volta-forever-%d" % os.geteuid())).is_symlink():
            (linked / ("volta-forever-%d" % os.geteuid())).symlink_to(self.root)
        code = "import sys\nfrom triton.backends.nvidia import compiler as c\nr = c.compile(sys.argv[1])\n" \
               "print(r[0].decode()[:11], r[2])\n"
        for cache, why in ((afile, "a-file: not a directory"),
                           (linked, "volta-forever-%d: it is a symbolic link" % os.geteuid())):
            r = self.run_vf([sys.executable, "-c", code, dot("f16")], self.env(PYTHONPATH=str(self.standin),
                                                                                TRITON_CACHE_DIR=str(cache),
                                                                                VF_LOG_LEVEL="2"))
            self.assertRegex(r.stdout.strip(), r"^cubin sm_80 ptxas-75-vf-tc", r.stderr)
            self.assertNotIn("VF_TRITON_TC is not used", r.stderr)
            self.assertEqual(r.stderr.count("are not counted in this process"), 1, r.stderr)
            self.assertIn(why, r.stderr)
        self.assertEqual(sorted(p.name for p in self.root.iterdir() if p.name.startswith(("slot-", "waiting-"))), [])

    def test_the_places_follow_tritons_cache_directory(self):
        # The directory of the places is that of Triton's cache when the process compiles its first
        # kernel that may be compiled for sm_80, not when it decides (at its first key): PyTorch's
        # Inductor, with TRITON_CACHE_DIR unset, sets it to its own cache's triton/0 between the two.
        home = self.root / "home-places"
        home.mkdir(exist_ok=True)
        later = self.root / "inductor-cache/triton/0"
        code = textwrap.dedent("""
            import json, os, sys
            from triton.runtime.cache import triton_key
            from triton.backends.nvidia import compiler as c
            key = triton_key()   # (the decision of the process, as Inductor's first graph key makes it)
            os.environ["TRITON_CACHE_DIR"] = sys.argv[2]
            print(json.dumps([key, c.compile(sys.argv[1])[0].decode()[:11]]))
            """)
        r = self.run_vf([sys.executable, "-c", code, dot("f16"), str(later)],
                        self.env(PYTHONPATH=str(self.standin), TRITON_CACHE_DIR=None, HOME=str(home)))
        key, target = self.last(r)
        self.assertEqual(target, "cubin sm_80", r.stderr)
        self.assertIn("-vf-tc", key)
        self.assertTrue(any(p.name.startswith("slot-") for p in self.own(later).iterdir()), r.stderr)
        self.assertFalse((home / ".triton").exists())

    def test_the_places_directory_replaced_by_a_link(self):
        # The directory of the places is opened once, without following a link, and used by its
        # descriptor: replaced by a link after that (by whoever may rename what is in Triton's cache
        # directory), the lock files are not made where the link points.
        cache = self.root / "cache-relinked"
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir(exist_ok=True)
        (elsewhere / "kept").write_text("not the layer's\n")
        own = self.own(cache)
        code = textwrap.dedent("""
            import json, sys
            from triton.backends.nvidia import compiler as c
            a = c.compile(sys.argv[1])[0].decode()[:11]
            for h in sys.modules["volta_forever_triton"].client()._the_pool().helpers:
                h.kill()
            print("compiled", flush=True)
            sys.stdin.readline()   # (the directory is replaced meanwhile)
            print(json.dumps([a, c.compile(sys.argv[2])[0].decode()[:11]]), flush=True)
            """)
        p = subprocess.Popen([str(self.vf), sys.executable, "-c", code, dot("f16"), dot("bf16")],
                             env=self.env(PYTHONPATH=str(self.standin), TRITON_CACHE_DIR=str(cache)),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(p.stdout.readline().strip(), "compiled")
            os.rename(own, str(own) + ".moved")
            os.symlink(elsewhere, own)
            out, err = p.communicate("go\n", timeout=60)
        finally:
            p.kill()
            p.wait()
        self.assertEqual(json.loads(out.strip().splitlines()[-1]), ["cubin sm_80"] * 2, err)
        self.assertEqual(sorted(os.listdir(elsewhere)), ["kept"])
        self.assertTrue(any(n.startswith("slot-") for n in os.listdir(str(own) + ".moved")))

    def test_a_stopped_helper(self):
        # A helper stopped from outside (SIGSTOP, a debugger, a frozen control group) past the time-out
        # is not taken for the kernel's: that kernel to Triton's own stages, not kept, and the next gets
        # a new helper. The time during which the program itself is stopped is not counted.
        code = textwrap.dedent("""
            import json, os, signal, sys, threading, time
            from triton.backends.nvidia import compiler as c
            out = []
            t = threading.Thread(target=lambda: out.append(c.compile(sys.argv[1])[0].decode()[:11]))
            t.start()
            vft = None
            while not vft or not vft._the_pool().helpers or not vft._the_pool().helpers[0].ready:
                time.sleep(0.02)
                vft = sys.modules["volta_forever_triton"]._state.get("client_module")
            time.sleep(0.3)
            stopped = vft._the_pool().helpers[0].pid
            os.kill(stopped, signal.SIGSTOP)
            t.join()
            out.append(c.compile(sys.argv[2])[0].decode()[:11])
            print(json.dumps([out, stopped]))
            """)
        r = self.run_vf([sys.executable, "-c", code, dot("f16", "SLEEP80\n"), dot("bf16")],
                        self.env(PYTHONPATH=str(self.standin), VF_TRITON_TC_TIMEOUT="2", TRITON_STANDIN_SLEEP="1"))
        out, stopped = self.last(r)
        self.assertEqual(out, ["cubin sm_75", "cubin sm_80"], r.stderr)
        self.assertIn("compiled for sm_75, not kept in Triton's cache: its helper was stopped from outside", r.stderr)
        self.assertTrue(self.gone([stopped]))
        # The program stopped for 4 s while its helper compiles for 6: with a time-out of 4 s, sm_80.
        code = textwrap.dedent("""
            import sys, time
            from triton.backends.nvidia import compiler as c
            c.compile(sys.argv[2])
            print("begin", flush=True)
            t = time.monotonic()
            print(c.compile(sys.argv[1])[0].decode()[:11], round(time.monotonic() - t), flush=True)
            """)
        p = subprocess.Popen([str(self.vf), sys.executable, "-c", code, dot("f16", "SLEEP80\n"), dot("i8")],
                             env=self.env(PYTHONPATH=str(self.standin), VF_TRITON_TC_TIMEOUT="4", TRITON_STANDIN_SLEEP="6"),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(p.stdout.readline().strip(), "begin")
            time.sleep(0.5)
            os.kill(p.pid, signal.SIGSTOP)
            time.sleep(4)
            os.kill(p.pid, signal.SIGCONT)
            out, err = p.communicate(timeout=60)
        finally:
            p.kill()
            p.wait()
        words = out.split()
        self.assertEqual(" ".join(words[:2]), "cubin sm_80", err)
        took = words[2]
        self.assertGreaterEqual(int(took), 6)

    def test_the_slots_bound_the_helpers(self):
        # The user's helpers on a host are VF_TRITON_TC_JOBS at most, whatever the number of processes
        # that compile at once: here 6 processes of 3 kernels each, 2 slots, every kernel for sm_80, and
        # never more than 2 helpers.
        e = self.env(PYTHONPATH=str(self.standin), VF_TRITON_TC_JOBS="2", TRITON_STANDIN_SLEEP="0.3")
        procs = [subprocess.Popen([str(self.vf), sys.executable, "-c", COMPILE] + [dot(t, "SLEEP80 // %d\n" % i)
                                  for t in ("f16", "bf16", "i8")], env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True) for i in range(6)]
        peak = 0
        while any(p.poll() is None for p in procs):
            peak = max(peak, len([pid for pid in helpers() if str(self.root).encode() in self.environ(pid)]))
            time.sleep(0.02)
        outs = [p.communicate() for p in procs]
        self.assertEqual([json.loads(o.strip().splitlines()[-1]) for o, _ in outs], [["cubin sm_80"] * 3] * 6,
                         outs[0][1])
        self.assertTrue(1 <= peak <= 2, peak)

    def environ(self, pid):
        try:
            with open("/proc/%d/environ" % pid, "rb") as f:
                return f.read()
        except OSError:
            return b""

    def test_no_free_slot(self):
        # The places all taken without end (here by another process's helpers, which compile kernels
        # that do not end, as stopped helpers or helpers stuck after their program died would hold
        # them): a process waits VF_TRITON_TC_WAIT seconds for a place once in its life, said once;
        # that kernel and every later one are then compiled for sm_80 by ALONE helper of its own
        # without a place, which two threads share, and no kernel waits for a place again.
        holder = textwrap.dedent("""
            import sys, threading, time
            from triton.backends.nvidia import compiler as c
            for t in sys.argv[1:]:
                threading.Thread(target=lambda t=t: c.compile(t), daemon=True).start()
            time.sleep(60)
            """)
        e = self.env(PYTHONPATH=str(self.standin), VF_TRITON_TC_JOBS="2", VF_TRITON_TC_WAIT="2",
                     VF_TRITON_TC_TIMEOUT="60", TRITON_STANDIN_SLEEP="0.5")
        p = subprocess.Popen([str(self.vf), sys.executable, "-c", holder, dot("f16", "HANG80\n"),
                              dot("bf16", "HANG80\n")], env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            end = time.monotonic() + 20
            while time.monotonic() < end and len([h for h in helpers() if self.environ(h).count(str(self.root).encode())]) < 2:
                time.sleep(0.05)
            code = textwrap.dedent("""
                import json, sys, threading, time
                from triton.backends.nvidia import compiler as c
                t = time.monotonic()
                out = [c.compile(x)[0].decode()[:11] for x in sys.argv[1:4]]
                first = round(time.monotonic() - t, 1)
                t = time.monotonic()
                ts = [threading.Thread(target=lambda x=x: out.append(c.compile(x)[0].decode()[:11])) for x in sys.argv[4:]]
                [x.start() for x in ts]; [x.join() for x in ts]
                v = sys.modules["volta_forever_triton"]
                print(json.dumps([out, first, round(time.monotonic() - t, 1), len(v.client()._the_pool().helpers),
                                  len(v._state.get("not_kept", ()))]))
                """)
            r = self.run_vf([sys.executable, "-c", code, dot("f16"), dot("i8"), dot("bf16"), dot("bf16", "SLEEP80\n"),
                             dot("f16", "SLEEP80 // 2\n")], e)
            out, first, threads, n, not_kept = self.last(r)
            self.assertEqual((out, n, not_kept), (["cubin sm_80"] * 5, 1, 0), r.stderr)
            self.assertTrue(2 <= first < 4.5, first)
            self.assertLess(threads, 1.9)   # (0.5 s each, one after the other in one helper: no wait for a place)
            self.assertEqual(r.stderr.count("no place for a helper of this host was free in 2 s"), 1, r.stderr)
            self.assertNotIn("not kept", r.stderr)
        finally:
            p.kill()
            p.wait()
        r = self.run_vf([sys.executable, "-c", COMPILE, dot("f16"), dot("i8")], e)
        self.assertEqual(self.last(r), ["cubin sm_80"] * 2, r.stderr)

    def test_threads_that_wait_for_a_place_together(self):
        # The places all taken without end, and four threads of one process that compile at once, as
        # PyTorch Inductor's main process precompiles its choices: they wait for a place together,
        # one wait of VF_TRITON_TC_WAIT seconds (said once); then the process's helper without a
        # place, started by one of them, compiles the four kernels one after the other (each of the
        # others waits for it as for a busy helper of its process), and none is left to Triton's
        # own stages.
        holder = textwrap.dedent("""
            import sys, threading, time
            from triton.backends.nvidia import compiler as c
            for t in sys.argv[1:]:
                threading.Thread(target=lambda t=t: c.compile(t), daemon=True).start()
            time.sleep(60)
            """)
        e = self.env(PYTHONPATH=str(self.standin), VF_TRITON_TC_JOBS="2", VF_TRITON_TC_WAIT="2",
                     VF_TRITON_TC_TIMEOUT="60", TRITON_STANDIN_SLEEP="0.5")
        p = subprocess.Popen([str(self.vf), sys.executable, "-c", holder, dot("f16", "HANG80\n"),
                              dot("bf16", "HANG80\n")], env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            end = time.monotonic() + 20
            while time.monotonic() < end and len([h for h in helpers() if self.environ(h).count(str(self.root).encode())]) < 2:
                time.sleep(0.05)
            code = textwrap.dedent("""
                import json, sys, threading, time
                from triton.backends.nvidia import compiler as c
                v = sys.modules["volta_forever_triton"]
                out, most = [], [0]
                def watch():
                    while True:
                        most[0] = max(most[0], len(v.client()._the_pool().helpers))
                        time.sleep(0.01)
                t = time.monotonic()
                ts = [threading.Thread(target=lambda x=x: out.append(c.compile(x)[0].decode()[:11])) for x in sys.argv[1:]]
                [x.start() for x in ts]
                threading.Thread(target=watch, daemon=True).start()
                [x.join() for x in ts]
                print(json.dumps([out, round(time.monotonic() - t, 1), most[0], len(v._state.get("not_kept", ()))]))
                """)
            r = self.run_vf([sys.executable, "-c", code] + [dot("f16", "SLEEP80 // %d\n" % i) for i in range(4)], e)
            out, took, most, not_kept = self.last(r)
            self.assertEqual((out, most, not_kept), (["cubin sm_80"] * 4, 1, 0), r.stderr)
            self.assertLess(took, 2 + 4 * 0.5 + 2.5)   # (one wait for a place, then the four one after the other)
            self.assertEqual(r.stderr.count("no place for a helper of this host was free in 2 s"), 1, r.stderr)
            self.assertNotIn("not kept", r.stderr)
        finally:
            p.kill()
            p.wait()

    def test_places_that_cannot_be_counted(self):
        # A lock file of the places that cannot be opened or locked for another reason than another
        # holder (flock failing with ENOSYS, as on Lustre without its flock option, or ENOLCK, as on
        # NFSv3 without a lock manager; the directory not writable; a directory at each lock file's
        # path) is not taken for a place that is taken: the places are not counted in that process
        # (said once, at VF_LOG_LEVEL=2), its helpers start without one, and no kernel waits.
        import socket
        host = __import__("re").sub(r"[^A-Za-z0-9_.]", "_", socket.gethostname())[:64]
        code = textwrap.dedent("""
            import errno, fcntl, json, os, sys, time
            how = sys.argv[1]
            if how in ("ENOSYS", "ENOLCK"):
                real = fcntl.flock
                def flock(fd, op):
                    if op & fcntl.LOCK_UN:
                        return real(fd, op)
                    raise OSError(getattr(errno, how), os.strerror(getattr(errno, how)))
                fcntl.flock = flock
            from triton.backends.nvidia import compiler as c
            t = time.monotonic()
            out = [c.compile(k)[0].decode()[:11] for k in sys.argv[2:]]
            print(json.dumps([out, round(time.monotonic() - t, 1)]))
            """)
        for how, why in (("ENOSYS", "Function not implemented"), ("ENOLCK", "No locks available"),
                         ("EACCES", "Permission denied"), ("EISDIR", "Is a directory")):
            cache = self.root / ("cache-places-" + how)
            own = self.own(cache)
            if how in ("EACCES", "EISDIR"):
                own.mkdir(parents=True, mode=0o700, exist_ok=True)
                if how == "EISDIR":
                    for name in ["slot-%s-%d" % (host, i) for i in range(2)] + ["waiting-" + host]:
                        (own / name).mkdir(exist_ok=True)
                os.chmod(own, 0o500 if how == "EACCES" else 0o700)
            try:
                for level in ("1", "2"):
                    r = self.run_vf([sys.executable, "-c", code, how, dot("f16"), dot("bf16"), dot("i8")],
                                    self.env(PYTHONPATH=str(self.standin), TRITON_CACHE_DIR=str(cache), VF_LOG_LEVEL=level,
                                             VF_TRITON_TC_JOBS="2", VF_TRITON_TC_WAIT="5"))
                    out, took = self.last(r)
                    self.assertEqual(out, ["cubin sm_80"] * 3, (how, r.stderr))
                    self.assertLess(took, 4.5, how)
                    lines = [x for x in r.stderr.splitlines() if "are not counted in this process" in x]
                    self.assertEqual(len(lines), 1 if level == "2" else 0, (how, r.stderr))
                    self.assertTrue(all(why in x for x in lines), (how, lines))
                    self.assertNotIn("no place", r.stderr)
            finally:
                if own.exists():
                    os.chmod(own, 0o700)

    def test_idle_helpers_give_their_slot_up(self):
        # A process whose helper is idle (its idle time not up) while another waits for the only slot:
        # the helper ends after a second, and the other process gets its kernel for sm_80 at once.
        e = self.env(PYTHONPATH=str(self.standin), VF_TRITON_TC_JOBS="1", VF_TRITON_TC_IDLE="60",
                     VF_TRITON_TC_WAIT="20")
        holder = ("import sys, time\nfrom triton.backends.nvidia import compiler as c\nc.compile(sys.argv[1])\n"
                  "print('compiled', flush=True)\ntime.sleep(30)\n")
        p = subprocess.Popen([str(self.vf), sys.executable, "-c", holder, dot("f16")], env=e, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True)
        try:
            self.assertEqual(p.stdout.readline().strip(), "compiled")
            mine = [h for h in helpers(p.pid)] or [h for h, pp in helpers().items() if b"vf-home" in self.environ(h)
                                                   and str(self.root).encode() in self.environ(h)]
            t = time.monotonic()
            r = self.run_vf([sys.executable, "-c", COMPILE, dot("bf16")], e)
            took = time.monotonic() - t
            self.assertEqual(self.last(r), ["cubin sm_80"], r.stderr)
            self.assertLess(took, 8)
            self.assertTrue(self.gone(mine), mine)
        finally:
            p.kill()
            p.wait()
            p.stdout.close()

    def test_100_processes(self):
        # 100 processes at once, two kernels each: every kernel for sm_80, never more helpers than the
        # slots (8), none left after.
        e = self.env(PYTHONPATH=str(self.standin), VF_TRITON_TC_JOBS="8")
        t = time.monotonic()
        procs = [subprocess.Popen([str(self.vf), sys.executable, "-c", COMPILE, dot("f16", "// %d\n" % i),
                                   dot("bf16", "// %d\n" % i)], env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True) for i in range(100)]
        peak = 0
        while any(p.poll() is None for p in procs):
            peak = max(peak, len([h for h in helpers() if str(self.root).encode() in self.environ(h)]))
            time.sleep(0.05)
        outs = [p.communicate() for p in procs]
        took = time.monotonic() - t
        bad = [err for o, err in outs if not o.strip() or json.loads(o.strip().splitlines()[-1]) != ["cubin sm_80"] * 2]
        self.assertEqual(bad, [])
        self.assertLessEqual(peak, 8)
        self.assertLess(took, 90)
        self.assertEqual([h for h in helpers() if str(self.root).encode() in self.environ(h)], [])

    def test_the_programs_limit_of_descriptors(self):
        # A program that has no descriptor left (its limit reached): that kernel to Triton's own
        # stages, not kept, with no traceback; once it has closed some, sm_80.
        code = textwrap.dedent("""
            import json, os, resource, sys, time
            from triton.backends.nvidia import compiler as c
            c.compile(sys.argv[3])
            time.sleep(2.5)   # (its helper ends)
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
            held = []
            try:
                while True:
                    held.append(os.open(os.devnull, os.O_RDONLY))
            except OSError:
                pass
            a = c.compile(sys.argv[1])[0].decode()[:11]
            for fd in held[-16:]:
                os.close(fd)
            b = c.compile(sys.argv[2])[0].decode()[:11]
            print(json.dumps([a, b]))
            """)
        r = self.run_vf([sys.executable, "-c", code, dot("f16"), dot("bf16"), dot("i8")],
                        self.env(PYTHONPATH=str(self.standin), VF_TRITON_TC_IDLE="1"))
        self.assertEqual(self.last(r), ["cubin sm_75", "cubin sm_80"], r.stderr)
        self.assertIn("VF_TRITON_TC: a helper could not be started: OSError: [Errno 24]", r.stderr)
        self.assertIn("not kept in Triton's cache, while none can be started", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_a_program_that_can_start_no_process(self):
        # A program that can start no process (RLIMIT_NPROC reached, as a pids control group's limit
        # gives): each kernel is compiled by Triton's own stages, not kept, and the layer says so
        # once at the default level (each kernel's line at VF_LOG_LEVEL=2).
        if os.geteuid() == 0:
            self.skipTest("RLIMIT_NPROC does not hold for root")
        code = textwrap.dedent("""
            import json, resource, sys
            from triton.backends.nvidia import compiler as c
            resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
            print(json.dumps([c.compile(t)[0].decode()[:11] for t in sys.argv[1:]]))
            """)
        kernels = [dot("f16"), dot("bf16"), dot("i8"), dot("f16", "// 2\n"), dot("f16", "// 3\n")]
        for level, lines in (("1", 1), ("2", 6)):
            r = self.run_vf([sys.executable, "-c", code] + kernels,
                            self.env(PYTHONPATH=str(self.standin), VF_LOG_LEVEL=level))
            self.assertEqual(self.last(r), ["cubin sm_75"] * 5, r.stderr)
            said = [x for x in r.stderr.splitlines() if "a helper could not be started: BlockingIOError" in x]
            self.assertEqual(len(said), lines, r.stderr)
            self.assertEqual(r.stderr.count("while none can be started"), 1, r.stderr)

    def test_a_program_that_closes_its_descriptors(self):
        # A program that closes every descriptor (its helper's socket with them) and opens others at
        # their numbers: its next kernel gets sm_80 from a new helper, the program's descriptors are
        # left as they are, and the old helper ends.
        code = textwrap.dedent("""
            import json, os, sys, time
            from triton.backends.nvidia import compiler as c
            a = c.compile(sys.argv[1])[0].decode()[:11]
            pool = sys.modules["volta_forever_triton"].client()._the_pool()
            old = [h.pid for h in pool.helpers]
            os.closerange(3, 4096)
            held = [os.open(os.devnull, os.O_RDONLY) for _ in range(20)]
            b = c.compile(sys.argv[2])[0].decode()[:11]
            ok = all(os.readlink("/proc/self/fd/%d" % fd) == os.devnull for fd in held)
            time.sleep(0.5)
            print(json.dumps([a, b, ok, [os.path.exists("/proc/%d" % p) for p in old]]))
            """)
        r = self.run_vf([sys.executable, "-c", code, dot("f16"), dot("bf16")], self.env(PYTHONPATH=str(self.standin)))
        self.assertEqual(self.last(r), ["cubin sm_80", "cubin sm_80", True, [False]], r.stderr)

    def test_a_program_that_is_a_subreaper(self):
        # A program that is a child subreaper (as Python as PID 1 of a container is) inherits the
        # ptxas of a helper killed while it runs, and of a compilation that does not end in its
        # time-out (the kill of the helper's group ends them): the helper's waiter reaps them, and
        # the program is left no zombie (PyPy, without waitid: not done).
        code = textwrap.dedent("""
            import ctypes, json, os, sys, time
            ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)   # PR_SET_CHILD_SUBREAPER
            from triton.backends.nvidia import compiler as c
            out = [c.compile(t)[0].decode()[:11] for t in sys.argv[1:]]
            time.sleep(1.5)
            zombies = []
            for p in os.listdir("/proc"):
                if p.isdigit():
                    try:
                        st = open("/proc/%s/stat" % p).read()
                    except OSError:
                        continue
                    name, rest = st.split("(", 1)[1].rsplit(")", 1)
                    if rest.split()[0] == "Z" and int(rest.split()[1]) == os.getpid():
                        zombies.append(name)
            print(json.dumps([out, zombies]))
            """)
        r = self.run_vf([sys.executable, "-c", code, dot("f16", "PTXKILLED80\n"), dot("bf16", "PTXAS80\n")],
                        self.env(PYTHONPATH=str(self.standin), VF_TRITON_TC_TIMEOUT="3"))
        out, zombies = self.last(r)
        self.assertEqual(out, ["cubin sm_75"] * 2, r.stderr)
        if hasattr(os, "waitid"):
            self.assertEqual(zombies, [], r.stderr)

    def test_a_program_that_execs(self):
        # A program that replaces itself with exec() while its helper lives: the helper's socket is
        # not inherited by the new image (close-on-exec), the helper sees its end and ends within a
        # second (a zombie of the new image, which does not know it, until that waits for any child
        # or ends).
        code = textwrap.dedent("""
            import os, sys
            from triton.backends.nvidia import compiler as c
            assert c.compile(sys.argv[1])[0].decode().startswith("cubin sm_80")
            helper = sys.modules["volta_forever_triton"].client()._the_pool().helpers[0].pid
            os.execv(sys.executable, [sys.executable, "-I", "-c", '''
            import json, os, sys, time
            helper, t = int(sys.argv[1]), time.monotonic()
            while time.monotonic() - t < 5:
                try:
                    st = open("/proc/%d/stat" % helper).read().rsplit(")", 1)[1].split()
                except OSError:
                    st = ["gone", None]
                if st[0] in ("Z", "gone"):
                    break
                time.sleep(0.01)
            print(json.dumps([st[0], st[1] == str(os.getpid()), round(time.monotonic() - t, 2)]))
            '''.replace("\\n            ", "\\n"), str(helper)])
            """)
        r = self.run_vf([sys.executable, "-c", code, dot("f16")], self.env(PYTHONPATH=str(self.standin)))
        state, parent, took = self.last(r)
        self.assertEqual((state, parent), ("Z", True), r.stderr)
        self.assertLess(took, 1.0)

    def test_a_fork_after_the_program_closed_its_descriptors(self):
        # A program that closed its descriptors (the layer's lock files of places that were taken
        # among them) and opened its own at their numbers, then forks: in the child, the program's
        # descriptors are left open (the layer closes there only those that are still its own).
        import fcntl
        import socket
        host = __import__("re").sub(r"[^A-Za-z0-9_.]", "_", socket.gethostname())[:64]
        cache = self.root / "cache-fork-fds"
        own = self.own(cache)
        own.mkdir(parents=True, mode=0o700, exist_ok=True)
        log = self.root / "program.log"
        held = []
        try:
            for i in range(7):   # (7 of 8 places taken by another holder: the pool keeps their files open)
                fd = os.open(str(own / ("slot-%s-%d" % (host, i))), os.O_RDWR | os.O_CREAT, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX)
                held.append(fd)
            code = textwrap.dedent("""
                import json, os, sys
                from triton.backends.nvidia import compiler as c
                a = c.compile(sys.argv[1])[0].decode()[:11]
                vft = sys.modules["volta_forever_triton"].client()
                pool = vft._the_pool()
                assert vft._slot(pool, vft._places()) is None   # (every place taken now, its helper's too)
                kept = sorted(fd for fd, _ in pool.files.values())
                os.closerange(3, 1024)
                log = os.open(sys.argv[2], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                r, w = os.pipe()
                mine = [log, w] + [os.open(os.devnull, os.O_RDONLY) for _ in range(10)]
                pid = os.fork()
                if pid == 0:
                    closed = []
                    for fd in mine:
                        try:
                            os.fstat(fd)
                        except OSError:
                            closed.append(fd)
                    try:
                        os.write(log, b"child\\n")
                        os.write(w, json.dumps(closed).encode())
                    finally:
                        os._exit(0)
                os.close(w)
                os.waitpid(pid, 0)
                print(json.dumps([a, kept, sorted(mine + [r]), os.read(r, 1000).decode()]))
                """)
            r = self.run_vf([sys.executable, "-c", code, dot("f16"), str(log)],
                            self.env(PYTHONPATH=str(self.standin), TRITON_CACHE_DIR=str(cache), VF_TRITON_TC_JOBS="8"))
            a, kept, mine, closed = self.last(r)
            self.assertEqual((a, len(kept)), ("cubin sm_80", 8), r.stderr)
            self.assertTrue(set(kept) & set(mine), (kept, mine))   # (the program's own at the layer's numbers)
            self.assertEqual(closed, "[]", r.stderr)
            self.assertEqual(log.read_text(), "child\n")
        finally:
            for fd in held:
                os.close(fd)

    def test_a_helper_killed_during_a_compilation(self):
        # The helper killed from outside (SIGKILL, as by the out-of-memory killer) while it compiles: the
        # kernel is compiled by Triton's own stages and not kept in Triton's cache, the next compilation
        # gets sm_80 from a new helper.
        code = textwrap.dedent("""
            import json, os, signal, sys, threading, time
            from triton.backends.nvidia import compiler as c
            c.compile(sys.argv[2])
            pool = sys.modules["volta_forever_triton"].client()._the_pool()
            pid = pool.helpers[0].pid
            out = []
            t = threading.Thread(target=lambda: out.append(c.compile(sys.argv[1])[0].decode()[:11]))
            t.start()
            time.sleep(1.5)
            os.kill(pid, signal.SIGKILL)
            t.join(30)
            out.append(c.compile(sys.argv[3])[0].decode()[:11])
            print(json.dumps(out))
            """)
        r = self.run_vf([sys.executable, "-c", code, dot("f16", "HANG80\n"), dot("bf16"), dot("i8")],
                        self.env(PYTHONPATH=str(self.standin), VF_TRITON_TC_TIMEOUT="60"))
        self.assertEqual(self.last(r), ["cubin sm_75", "cubin sm_80"], r.stderr)
        self.assertIn("not kept in Triton's cache: its sm_80 compilation was ended from outside (signal 9)", r.stderr)

    def test_what_a_request_may_ask(self):
        # A helper checks each value of a request, not only its names: ptxas's options of the ones
        # Triton gives (no --options-file, no path), libraries of Triton's directory alone, no IR given
        # in place of a stage, other values plain (no line end: fullmatch), texts bounded before a path
        # is resolved (a path of 7 MB answered at once); a request is refused at once, and the helper
        # goes on serving; JSON nested too deep for the parser is refused, not a crash; a message over
        # 64 MiB ends the helper (its next kernel gets a new one).
        secret = self.root / "secret.txt"
        secret.write_text("secret-line-from-a-file\n")
        os.chmod(secret, 0o600)
        code = textwrap.dedent("""
            import dataclasses, json, struct, sys, time
            from triton.backends.nvidia import compiler as c
            first = c.compile(sys.argv[1])[0].decode()[:11]
            vft = sys.modules["volta_forever_triton"].client()
            base = dataclasses.asdict(c.CUDABackend(c.GPUTarget("cuda", 75, 32)).parse_options({}))
            def request(ttir, **options):
                r = {"kind": "compile", "ttir": ttir, "target": {"backend": "cuda", "arch": 75, "warp_size": 32},
                     "options": dict(base, **options), "metadata": {}}
                r["digest"] = vft._digest(r)
                return r
            h = vft._take()
            answers = []
            try:
                for r in (request(sys.argv[2], ptx_options="--options-file " + sys.argv[3]),
                          request(sys.argv[2], extern_libs=[["libdevice", sys.argv[3]]]),
                          request(sys.argv[2], ir_override="/x.ttgir"),
                          request(sys.argv[2], backend_name="../../x"),
                          request(sys.argv[2], backend_name="cuda\\n"),
                          request(sys.argv[2], extern_libs=[["libdevice", "/tmp/.." * 1000000]]),
                          dict(request(sys.argv[2]), digest="0" * 64),
                          request(sys.argv[2], ptx_options="-v --maxrregcount=64 --opt-level 0")):
                    t = time.monotonic()
                    a = h.ask(r, 30)
                    answers.append([a["kind"], a.get("why", "")[:120], round(time.monotonic() - t, 1)])
                vft._send(h.sock, struct.pack("<Q", 100000) + b"[" * 100000)
                answers.append([h.receive(30)["kind"]])
                vft._send(h.sock, struct.pack("<Q", 100 << 20))
                try:
                    ended = h.receive(10) is None and False
                except EOFError:
                    ended = True
            finally:
                vft._give(h)
            print(json.dumps([first, answers, ended, c.compile(sys.argv[4])[0].decode()[:11]]))
            """)
        r = self.run_vf([sys.executable, "-c", code, dot("f16"), dot("bf16"), str(secret), dot("i8")],
                        self.env(PYTHONPATH=str(self.standin)))
        first, answers, ended, after = self.last(r)
        self.assertEqual((first, ended, after), ("cubin sm_80", True, "cubin sm_80"), r.stderr)
        wants = ("ptx_options give ptxas '--options-file'", "extern_libs name %s" % secret, "gives an IR in place of a stage",
                 "a value of backend_name", "a value of backend_name", "extern_libs are not pairs",
                 "a request whose digest is not that of what it asks")
        for (kind, why, took), want in zip(answers, wants):
            self.assertEqual(kind, "invalid", (want, why))
            self.assertIn(want, why)
            self.assertLess(took, 2)
        self.assertEqual(answers[7][0], "ok", answers[7])
        self.assertEqual(answers[8], ["invalid"])
        self.assertNotIn("secret-line", r.stdout + r.stderr)

    def test_answers_that_are_refused(self):
        # What a helper answers is checked before Triton or its cache takes it (against a bug, or a
        # Triton that makes other things than the layer expects): a cubin that is not ELF, one for
        # another SM, one whose kept PTX is not the answer's PTX, a metadata that changes a field of the
        # request; each: Triton's own stages, said once, and the process compiles as without the switch.
        for mark, why in (("NOTELF80", "it is not a cubin (no 64-bit ELF header)"),
                          ("SM75CUBIN80", "its cubin is for sm_75, not sm_80"),
                          ("TAMPER80", "the PTX that its cubin keeps is not its PTX"),
                          ("BADMD80", "it changes the metadata's debug")):
            code = textwrap.dedent("""
                import json, sys
                from triton.backends.nvidia import compiler as c
                out = [c.compile(t) for t in sys.argv[1:]]
                print(json.dumps([[o[0].decode()[:11] for o in out], [o[2] for o in out]]))
                """)
            r = self.run_vf([sys.executable, "-c", code, dot("f16", mark + "\n"), dot("bf16")],
                            self.env(PYTHONPATH=str(self.standin)))
            out, keys = self.last(r)
            self.assertEqual(out, ["cubin sm_75"] * 2, (mark, r.stderr))
            self.assertEqual(keys[1], "ptxas-75", mark)
            self.assertEqual(r.stderr.count("its helper's answer was refused: " + why), 1, (mark, r.stderr))
            self.assertEqual(r.stderr.count("every kernel of this process is compiled by Triton's own stages"), 1, mark)

    def test_a_link_where_the_cubin_is(self):
        # A link where ptxas writes the cubin is not followed: the compilation fails at once, and the
        # kernel goes to Triton's own stages, not kept.
        r = self.compile(dot("f16", "LINK80\n"))
        self.assertTarget(r, 75)
        self.assertIn("not kept in Triton's cache: OSError: [Errno 40]", r[4])

    def test_a_work_directory_swapped(self):
        # The helper's directory renamed away and another put at its path while it compiles (what
        # only someone who may already change Triton's cache can do): the cubin is read through the
        # descriptor of the helper's own directory, not by its path, so that what ptxas wrote in the
        # other directory is not taken; the kernel goes to Triton's own stages, not kept.
        r = self.compile(dot("f16", "SWAP80\n"))
        self.assertTarget(r, 75)
        self.assertIn("not kept in Triton's cache: FileNotFoundError", r[4])
        self.assertNotIn("forged", r[0])


class Module(Base):
    """volta_forever_triton of the test's installation, read in this process."""

    def module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("vft_under_test", str(self.home / "python/volta_forever_triton.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules["vft_under_test"] = m
        spec.loader.exec_module(m)
        return m

    def test_the_rule_of_slower_products(self):
        # slower(): a kernel is left to 7.5 when every one of its products matches a kind of SLOWER:
        # types, accumulator, ranges of the tile (TTIR) and how B is stored (the sm_80 TTGIR); one
        # product that does not match keeps it for sm_80. b_storage() reads the order of the shared
        # memory from which B is loaded, as Triton 3.4 to 3.9 write it and as Triton 3.3 does.
        m = self.module()
        m.SLOWER = ({"why": "rule-x", "types": ("bf16",), "acc": ("f32",), "m": (1, 16), "n": (128, 1 << 30),
                     "elements": (1, 16 * 128 * 64), "b": "n"},)

        def kernel(*dots):
            body = "".join("  %%c%d = tt.dot %%x, %%y, %%acc : tensor<%dx%dx%s> * tensor<%dx%dx%s> -> tensor<%dx%dx%s>\n"
                           % (i, M, K, t, K, N, t, M, N, r) for i, (t, r, M, N, K) in enumerate(dots))
            return "tt.func public @k() {\n" + body + "  tt.return\n}\n"
        rows, cols = TTGIR_B.format(order="1, 0"), TTGIR_B.format(order="0, 1")
        self.assertEqual(m.b_storage(rows), "n")
        self.assertEqual(m.b_storage(cols), "k")
        self.assertEqual(m.b_storage(TTGIR_B33), "n")
        self.assertIsNone(m.b_storage(rows + cols.replace("#shared", "#shared9")))   # mixed
        self.assertIsNone(m.b_storage("tt.func public @k() {\n}\n"))                 # no B from shared memory
        self.assertEqual(m._products(kernel(("bf16", "f32", 16, 128, 64))), [("bf16", "f32", 16, 128, 64)])
        self.assertIn("rule-x", m.slower(kernel(("bf16", "f32", 16, 128, 64)), rows))
        self.assertIn("rule-x", m.slower(kernel(("bf16", "f32", 16, 256, 32)), rows))
        for ttir, ttgir in ((kernel(("bf16", "f32", 16, 128, 64)), cols), (kernel(("bf16", "f32", 32, 128, 64)), rows),
                            (kernel(("bf16", "f32", 16, 64, 64)), rows), (kernel(("f16", "f32", 16, 128, 64)), rows),
                            (kernel(("bf16", "f32", 16, 128, 128)), rows),
                            (kernel(("bf16", "f32", 16, 128, 64), ("bf16", "f32", 64, 128, 64)), rows)):
            self.assertIsNone(m.slower(ttir, ttgir), ttir)
        m.SLOWER = ()
        self.assertIsNone(m.slower(kernel(("bf16", "f32", 16, 128, 64)), rows))

    def test_slower_products_with_triton_33(self):
        # Triton 3.3 compiles no product of halves for 7.5 (its fma path fails in make_llir): a kernel
        # of a kind of SLOWER is compiled for sm_80 all the same, the only way it compiles. Triton 3.4
        # and later: left to 7.5.
        import types
        m = self.module()
        ttir = ("tt.func public @k() {\n  %c = tt.dot %x, %y, %acc : tensor<16x32xbf16> * tensor<32x128xbf16> -> "
                "tensor<16x128xf32>\n  tt.return\n}\n")
        rows = TTGIR_B.format(order="1, 0")
        saved = sys.modules.get("triton")
        try:
            seen = []
            for version in ("3.3.1", "3.4.0", "3.9.0"):
                sys.modules["triton"] = types.SimpleNamespace(__version__=version)
                seen.append(m.slower(ttir, rows) is not None)
        finally:
            if saved is None:
                sys.modules.pop("triton", None)
            else:
                sys.modules["triton"] = saved
        self.assertEqual(seen, [False, True, True])


    @staticmethod
    def cubin(sm, ptx, body=b"cubin\n"):
        """A cubin as the stand-in's make_cubin makes it: an ELF header, a table of sections, and the
        section .nv_debug_ptx_txt with the PTX, its lines ended by NUL, as ptxas keeps it."""
        import struct
        names = b"\0.shstrtab\0" + (b".nv_debug_ptx_txt\0" if ptx is not None else b"")
        data = b"\0\0" + names
        rows = [bytes(64), struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, 64 + len(body) + 2, len(names), 0, 0, 1, 0)]
        if ptx is not None:
            text = ptx.replace("\n", "\0").encode()
            rows.append(struct.pack("<IIQQQQIIQQ", 11, 1, 0, 0, 64 + len(body) + len(data), len(text), 0, 0, 1, 0))
            data += text
        head = (b"\x7fELF" + bytes([2, 1, 1, 0x33, 7]) + bytes(7)
                + struct.pack("<HHIQQQIHHHHHH", 2, 190, 1, 0, 0, 64 + len(body) + len(data), sm, 64, 0, 0, 64, len(rows), 1))
        return head + body + data + b"".join(rows)

    def test_the_checks_of_an_answer(self):
        # The cubin of an answer: an ELF64 of machine EM_CUDA for sm_80 that keeps its PTX, and the PTX
        # it keeps the answer's, compared as ptxas keeps it (no comments, no .loc or .file lines, no
        # DWARF sections, blanks as one space); a PTX that differs in one constant is refused; a table
        # of sections that is not one is refused, never an exception.
        c = self.module().client()
        ptx = ("//\n// Generated by LLVM NVPTX Back-End\n//\n\n.version 8.7\n.target sm_80\n.address_size 64\n\n"
               "\t// .globl\tk\n.visible .entry k(\n\t.param .u64 k_param_0\n)\n{\n\t.loc\t1 7 0\n"
               "\tmov.b32 \t%r1, 0f00000000;   // the start\n\tret;\n}\n\t.file\t1 \"/x/k.py\"\n"
               "\t.section\t.debug_abbrev\n\t{\n.b8 1\n\t}\n")
        kept = ".version 8.7\n.target sm_80\n.address_size 64\n.visible .entry k(\n.param .u64 k_param_0\n)\n{\n" \
               "mov.b32 %r1, 0f00000000;\nret;\n}\n"
        honest = self.cubin(80, kept)
        self.assertIsNone(c.cubin_problem(honest))
        self.assertIsNone(c.ptx_problem(honest, ptx))
        self.assertEqual(c.kept_ptx(honest).split("\n")[:2], [".version 8.7", ".target sm_80"])
        tampered = self.cubin(80, kept.replace("0f00000000", "0f3F800000"))
        self.assertIn("the PTX that its cubin keeps is not its PTX (line 8", c.ptx_problem(tampered, ptx))
        self.assertIn("has 9 lines, its PTX 10", c.ptx_problem(self.cubin(80, kept[:-2]), ptx))
        self.assertEqual(c.cubin_problem(self.cubin(75, kept)), "its cubin is for sm_75, not sm_80")
        self.assertEqual(c.cubin_problem(self.cubin(80, None)), "its cubin does not keep its PTX")
        self.assertEqual(c.ptx_problem(self.cubin(80, None) + b".nv_debug_ptx_txt", ptx), "its cubin's PTX could not be read")
        broken = bytearray(honest)
        broken[0x28:0x30] = (len(honest) - 10).to_bytes(8, "little")   # (the table of sections out of the file)
        self.assertEqual(c.ptx_problem(bytes(broken), ptx), "its cubin's PTX could not be read")
        import random
        rng = random.Random(7)
        for _ in range(200):
            junk = bytearray(honest)
            for _ in range(8):
                junk[rng.randrange(len(junk))] = rng.randrange(256)
            c.ptx_problem(bytes(junk), ptx)   # (any answer, no exception)
        self.assertEqual(c.cubin_problem(b"\x7fELF" + bytes(60)), "it is not a cubin (no 64-bit ELF header)")

    def test_messages(self):
        # A message (8 bytes of size, then JSON) is read as it comes, a limit on its size; JSON that is
        # nested too deep for the parser, or not UTF-8, or not an object, is no message (not an error);
        # the digest of a request covers its TTIR, target, options and metadata.
        import socket
        import struct
        c = self.module().client()
        a, b = socket.socketpair()
        try:
            b.setblocking(False)
            inbox = c._Inbox(b, 1000)
            data = c._message({"kind": "x", "n": 1})
            a.sendall(data[:5])
            self.assertIsNone(inbox.read())
            a.sendall(data[5:])
            self.assertEqual(c._decode(inbox.read()), {"kind": "x", "n": 1})
            a.sendall(struct.pack("<Q", 1001))
            with self.assertRaises(ValueError):
                inbox.read()
            a.close()
            with self.assertRaises(EOFError):
                c._Inbox(b, 10).read()
        finally:
            a.close()
            b.close()
        for bad in (b"[" * 100000, b"{\"a\":" * 50000, b"\xff\xfe", b"[1, 2]", b""):
            self.assertIsNone(c._decode(bad))
        r = {"ttir": "t", "target": {"arch": 75}, "options": {"num_warps": 4}, "metadata": {}}
        d = c._digest(r)
        self.assertEqual(d, c._digest(dict(r, kind="compile", digest="x")))
        self.assertNotEqual(d, c._digest(dict(r, options={"num_warps": 8})))

    def test_the_values_a_request_may_have(self):
        # Of the helper's checks: the texts of a request bounded before a path is resolved (a path of
        # 7 MB, of 2,000,000 components, refused at once), values matched whole (a line end is not
        # taken), ptxas's options of Triton's alone.
        import importlib.util
        spec = importlib.util.spec_from_file_location("vft_helper_under_test",
                                                      str(self.home / "python/volta_forever_triton_helper.py"))
        h = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(h)
        import dataclasses

        @dataclasses.dataclass
        class Options:
            num_warps: int = 4
            num_stages: int = 3
            backend_name: str = "cuda"
            ptx_options: str = None
            extern_libs: tuple = None

        def request(**options):
            return {"ttir": "t", "target": {"backend": "cuda", "arch": 75, "warp_size": 32}, "options": options,
                    "metadata": {}}
        t = time.monotonic()
        self.assertIn("not pairs", h._request_problem(request(extern_libs=[["x", "/tmp/.." * 1000000]]), Options,
                                                      str(self.root), ()))
        self.assertLess(time.monotonic() - t, 1)
        for value in ("cuda\n", "cuda\x00", "a/b", "x" * 65):
            self.assertIn("a value of backend_name", h._request_problem(request(backend_name=value), Options,
                                                                       str(self.root), ()) or "", repr(value))
        self.assertIsNone(h._request_problem(request(backend_name="cuda", num_warps=8), Options, str(self.root), ()))
        self.assertIn("ptx_options", h._request_problem(request(ptx_options="-v " * 2000), Options, str(self.root), ()))
        self.assertIn("'--options-file'", h._request_problem(request(ptx_options="-v --options-file /x"), Options,
                                                             str(self.root), ()))
        self.assertEqual(h._request_problem(request(num_warps=4096), Options, str(self.root), ()),
                         "numbers of warps, stages or registers out of their ranges")

    def test_a_helper_reaps_its_children(self):
        # A helper that ends otherwise than killed kills what it still runs (a ptxas left by a
        # compilation that failed) and reaps its children, a second at most: none is left to its
        # program or to a subreaper.
        code = textwrap.dedent("""
            import importlib.util, json, subprocess, sys, time
            spec = importlib.util.spec_from_file_location("vft_helper_children", sys.argv[1])
            h = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(h)
            kids = [subprocess.Popen(["sleep", "41.25"]).pid for _ in range(2)] + [subprocess.Popen(["true"]).pid]
            time.sleep(0.3)   # (the last one has ended: a zombie until it is reaped)
            t = time.monotonic()
            h._end_children()
            states = []
            for pid in kids:
                try:
                    states.append(open("/proc/%d/stat" % pid).read().rsplit(")", 1)[1].split()[0])
                except OSError:
                    states.append("gone")
            print(json.dumps([states, round(time.monotonic() - t, 1)]))
            """)
        r = subprocess.run([sys.executable, "-c", code, str(self.home / "python/volta_forever_triton_helper.py")],
                           capture_output=True, text=True, timeout=60)
        self.assertTrue(r.stdout.strip(), r.stderr)
        states, took = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(states, ["gone"] * 3, r.stderr)
        self.assertLess(took, 1.5)

    def test_a_send_to_a_closed_peer(self):
        # A message sent on a socket whose peer has closed (a helper that ended at its idle time as
        # the request was sent) gives an error, not SIGPIPE, which would end a program that sets it
        # to its default: in such a program, the error and status 0.
        code = textwrap.dedent("""
            import importlib.util, signal, socket, sys
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)
            spec = importlib.util.spec_from_file_location("vft_pipe", sys.argv[1])
            m = importlib.util.module_from_spec(spec)
            sys.modules["vft_pipe"] = m
            spec.loader.exec_module(m)
            a, b = socket.socketpair()
            b.close()
            try:
                m.client()._send(a, m.client()._message({"kind": "hello"}))
            except OSError as e:
                print(type(e).__name__)
            """)
        r = subprocess.run([sys.executable, "-c", code, str(self.home / "python/volta_forever_triton.py")],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "BrokenPipeError"), r.stderr)

    def test_gpu_names(self):
        # VF_TRITON_TC_GPUS (the tests') stands for the driver's names only on a machine that has
        # no NVIDIA driver: it cannot make a machine with another GPU pass for all Voltas.
        m = self.module()
        root = self.root / "proc-gpus"
        (root / "0000:18:00.0").mkdir(parents=True, exist_ok=True)
        (root / "0000:18:00.0/information").write_text("Model: \t\t Tesla T4\nIRQ: 1\n")
        saved = os.environ.get("VF_TRITON_TC_GPUS")
        os.environ["VF_TRITON_TC_GPUS"] = VOLTA
        try:
            self.assertEqual(m._gpu_names(str(root)), ["Tesla T4"])
            # A driver's directory without the GPUs' (no GPU): none; no driver at all: the given names.
            (self.root / "driver-no-gpus").mkdir(exist_ok=True)
            self.assertEqual(m._gpu_names(str(self.root / "driver-no-gpus/gpus")), [])
            self.assertEqual(m._gpu_names(str(self.root / "no-driver/gpus")), [VOLTA])
        finally:
            if saved is None:
                os.environ.pop("VF_TRITON_TC_GPUS")
            else:
                os.environ["VF_TRITON_TC_GPUS"] = saved

    def test_tool_of_its_installation(self):
        # The tool beside the module, not that of VOLTA_FOREVER_HOME when that is another
        # installation (a launcher of another release that started this Python).
        other = self.root / "other-home"
        (other / "bin").mkdir(parents=True, exist_ok=True)
        (other / "bin/vf-translate").write_text("#!/bin/sh\nexit 0\n")
        (other / "bin/vf-translate").chmod(0o755)
        m = self.module()
        saved = os.environ.get("VOLTA_FOREVER_HOME")
        os.environ["VOLTA_FOREVER_HOME"] = str(other)
        try:
            self.assertEqual(m._tool(), str(self.home / "bin/vf-translate"))
        finally:
            if saved is None:
                os.environ.pop("VOLTA_FOREVER_HOME")
            else:
                os.environ["VOLTA_FOREVER_HOME"] = saved

    def test_layer_takes_adapt_worker_reuses_process_and_stops(self):
        import importlib.util
        tool = ROOT / "dist/bin/vf-translate"
        if not tool.is_file():
            self.skipTest("vf-translate not built")
        # Use standalone helper copy pointing at dist/bin/vf-translate.
        (self.home / "bin/vf-translate").unlink(missing_ok=True)
        (self.home / "bin/vf-translate").symlink_to(tool)
        spec = importlib.util.spec_from_file_location("vft_helper_worker", str(self.home / "python/volta_forever_triton_helper.py"))
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        bad_img = b".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry bad() { barrier.cluster.arrive; ret; }\n"
        try:
            proc = helper._ensure_adapt_worker(str(self.home / "bin/vf-translate"), dict(os.environ))
            self.assertIsNotNone(proc)
            pid = proc.pid
            r1 = helper.layer_takes(bad_img)
            self.assertIsNotNone(r1[0])
            self.assertFalse(r1[1])
            self.assertIs(helper._adapt_worker["proc"], proc)
            r2 = helper.layer_takes(bad_img)
            self.assertEqual(r1, r2)
            self.assertEqual(helper._adapt_worker["proc"].pid, pid)
        finally:
            helper._end_children()
        self.assertIsNone(helper._adapt_worker["proc"])

    def test_layer_takes_worker_matches_oneshot_parity(self):
        import importlib.util
        tool = ROOT / "dist/bin/vf-translate"
        if not tool.is_file():
            self.skipTest("vf-translate not built")
        (self.home / "bin/vf-translate").unlink(missing_ok=True)
        (self.home / "bin/vf-translate").symlink_to(tool)
        spec = importlib.util.spec_from_file_location("vft_helper_parity", str(self.home / "python/volta_forever_triton_helper.py"))
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        ptx_ok = b".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry k() { ret; }\n"
        ptx_75 = b".version 8.0\n.target sm_75\n.address_size 64\n.visible .entry k() { ret; }\n"
        ptx_70 = b".version 8.0\n.target sm_70\n.address_size 64\n.visible .entry k() { ret; }\n"
        ptx_trap_and_refused = b".version 8.8\n.target sm_80\n.address_size 64\n.visible .entry bad() { barrier.cluster.arrive; ret; }\n"
        sm75_cubin = (ROOT / "tests/kernels/translate/k_memory.sm75.cubin").read_bytes()
        clean_env = {k: v for k, v in os.environ.items() if k not in ("VF_LOG", "VF_LOG_LEVEL", "LD_AUDIT", "LD_PRELOAD")}
        cases = [
            ("empty", b""),
            ("corrupt_elf", b"\x7fELFgarbage"),
            ("plain_text", b"not ptx or elf"),
            ("ptx_sm70", ptx_70),
            ("ptx_sm75", ptx_75),
            ("ptx_sm80_ok", ptx_ok),
            ("ptx_sm80_trap_precedence", ptx_trap_and_refused),
            ("cubin_sm75", sm75_cubin),
        ]
        try:
            proc = helper._ensure_adapt_worker(str(self.home / "bin/vf-translate"), clean_env)
            self.assertIsNotNone(proc)
            pid = proc.pid
            for name, payload in cases:
                oneshot = helper._layer_takes_oneshot(str(self.home / "bin/vf-translate"), clean_env, payload)
                worker = helper.layer_takes(payload)
                self.assertEqual(worker[0] is None, oneshot[0] is None, f"{name}: {worker} vs {oneshot}")
                self.assertEqual(worker[1], oneshot[1], f"{name}: transient mismatch {worker} vs {oneshot}")
                if oneshot[0] is not None:
                    self.assertEqual(worker[0].split(":", 1)[0], oneshot[0].split(":", 1)[0], f"{name}: {worker} vs {oneshot}")
                    if oneshot[0].startswith("the layer would not run it all: "):
                        self.assertEqual(worker[0], oneshot[0], f"{name}: {worker} vs {oneshot}")
                self.assertIsNotNone(helper._adapt_worker["proc"], f"{name}: worker died")
                self.assertEqual(helper._adapt_worker["proc"].pid, pid, f"{name}: worker restarted")
        finally:
            helper._end_children()
        self.assertIsNone(helper._adapt_worker["proc"])




@unittest.skipUnless(os.environ.get("VF_TRITON_PYTHON"), "VF_TRITON_PYTHON (a Python with Triton) is not set")
class RealTriton(unittest.TestCase):
    """Through the launcher of this tree (its proxy, its offline tool): what Triton makes."""

    @classmethod
    def setUpClass(cls):
        if not (ROOT / "dist/bin/vf-translate").is_file() or not PROXY.is_file():
            raise unittest.SkipTest("the layer is not built (scripts/build.sh)")
        cls.scratch = tempfile.TemporaryDirectory(prefix="vf-triton-tc-real-")
        cls.root = Path(cls.scratch.name)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def compile(self, kernel, tc="1", cache=None, extra=None, *args):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("VF_", "TRITON_", "VOLTA_FOREVER"))}
        env.update(VF_QUIET="1", VF_TRITON_TC=tc, VF_TRITON_TC_GPUS=VOLTA, HOME=str(self.root), TMPDIR=str(self.root),
                   PYTHONDONTWRITEBYTECODE="1", TRITON_CACHE_DIR=str(cache or self.root / ("cache-" + kernel + tc)),
                   VF_LOG=str(self.root / "vf.log"), VF_LOG_LEVEL="2")
        env.update(extra or {})
        r = subprocess.run([str(ROOT / "scripts/vf"), os.environ["VF_TRITON_PYTHON"],
                            str(ROOT / "tests/python/triton_tc_compile.py"), kernel] + list(args),
                           env=env, capture_output=True, text=True, timeout=600)
        return r

    def made(self, *a, **kw):
        r = self.compile(*a, **kw)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def alone(self, kernel, *args):
        """What Triton makes of the kernel with the switch off; None when it cannot compile it for
        7.5 (Triton 3.3 cannot convert float16 to float32 in its fma path; int8 products fail in
        Triton 3.8's fma path): that is Triton's, as without the layer."""
        r = self.compile(kernel, "0", None, None, *args)
        if r.returncode != 0:
            sys.stderr.write("\n  (Triton alone cannot compile %s %s for 7.5: %s)\n" % (
                kernel, " ".join(args), (r.stderr.strip().splitlines() or ["?"])[-1][:200]))
            return None
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_kernels(self):
        for kernel in ("mm-f16", "mm-bf16", "mm-i8"):
            off = self.alone(kernel)
            k = self.made(kernel)
            self.assertEqual((k["target"], k["keeps_ptx"]), ("sm_80", True), kernel)
            self.assertGreater(k["mma"], 0, kernel)
            if off is not None:
                self.assertEqual((off["target"], off["mma"]), ("sm_75", 0), kernel)
        for kernel in ("mm-f32", "add"):
            off = self.alone(kernel)
            if off is None:
                self.assertNotEqual(self.compile(kernel).returncode, 0, kernel)
                continue
            on = self.made(kernel)
            self.assertEqual((on["target"], on["mma"]), ("sm_75", 0), kernel)
            self.assertEqual(on["cubin"], off["cubin"], kernel)

    def test_the_cubin_of_a_compilation_in_the_process(self):
        # A helper's cubin is that of the same sm_80 compilation in the process (compile80 of
        # python/volta_forever_triton_helper.py on the kernel's TTIR for 7.5), byte for byte, for each
        # type, B stored either way, and a block made with fewer stages for its shared memory.
        for kernel, args in (("mm-f16", ()), ("mm-bf16", ()), ("mm-i8", ()), ("mmt-f16", ()),
                             ("mmt-bf16", ("--block", "128x128x32", "--warps", "8")),
                             ("mm-f16", ("--block", "128x256x64", "--warps", "8", "--stages", "4"))):
            on = self.made(kernel, "1", None, None, *args)
            r = self.compile(kernel, "0", None, None, *(args + ("--inproc",)))
            if r.returncode != 0 and on["triton"].startswith("3.3."):
                continue   # (Triton 3.3 cannot compile the product for 7.5, whose TTIR this takes)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            alone = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertEqual((on["target"], on["cubin"]), ("sm_80", alone["cubin"]), (kernel, args))

    def test_cache(self):
        cache = self.root / "one-cache"
        kernel = "mm-f16" if self.alone("mm-f16") is not None else "mm-bf16"
        if self.alone(kernel) is None:
            self.skipTest("Triton alone compiles no product of halves for 7.5")
        a = self.made(kernel, "0", cache)
        b = self.made(kernel, "1", cache)
        c = self.made(kernel, "0", cache)
        d = self.made(kernel, "1", cache)
        self.assertEqual([k["target"] for k in (a, b, c, d)], ["sm_75", "sm_80", "sm_75", "sm_80"])
        self.assertEqual([len(k["cache"]) for k in (a, b, c, d)], [1, 2, 2, 2])
        self.assertEqual((a["cubin"], b["cubin"]), (c["cubin"], d["cubin"]))
        # The same cache with VF_TRANSLATE=first (the layer would take the sm_80 machine code, not its
        # PTX): what was made with the switch is not taken; with VF_PAIRS=0, made again.
        e = self.made(kernel, "1", cache, {"VF_TRANSLATE": "first"})
        f = self.made(kernel, "1", cache, {"VF_PAIRS": "0"})
        self.assertEqual((e["target"], e["cubin"], f["target"], len(f["cache"])), ("sm_75", a["cubin"], "sm_80", 3))

    def test_a_transient_failure_is_not_kept(self):
        # The layer's tool fails (it could not run: a full disk, a time-out): the kernel is
        # compiled for 7.5 and given to the program, but not left in Triton's cache under the mark,
        # where the next compilation would take it for good.
        if self.alone("mm-f16") is None:
            self.skipTest("Triton alone cannot compile this kernel for 7.5 (Triton 3.3)")
        home = self.root / "home-broken-tool"
        if not home.exists():
            (home / "bin").mkdir(parents=True)
            (home / "lib").mkdir()
            shutil.copy2(ROOT / "scripts/vf", home / "bin/vf")
            shutil.copytree(ROOT / "python", home / "python", ignore=shutil.ignore_patterns("__pycache__"))
            (home / "lib/libcuda.so.1").symlink_to(PROXY)
            (home / "bin/vf-translate").write_text("#!/bin/sh\necho 'no space left on device' >&2\nexit 1\n")
            (home / "bin/vf-translate").chmod(0o755)
        cache = self.root / "cache-transient"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("VF_", "TRITON_", "VOLTA_FOREVER"))}
        env.update(VF_QUIET="1", VF_TRITON_TC="1", VF_TRITON_TC_GPUS=VOLTA, HOME=str(self.root), TMPDIR=str(self.root),
                   PYTHONDONTWRITEBYTECODE="1", TRITON_CACHE_DIR=str(cache), VF_LOG=str(self.root / "vf-t.log"))
        r = subprocess.run([str(home / "bin/vf"), os.environ["VF_TRITON_PYTHON"],
                            str(ROOT / "tests/python/triton_tc_compile.py"), "mm-f16"], env=env, capture_output=True,
                           text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1])["target"], "sm_75")
        self.assertIn("no space left on device", (self.root / "vf-t.log").read_text())
        k = self.made("mm-f16", "1", cache)
        self.assertEqual(k["target"], "sm_80")

    def test_shared_memory(self):
        block = ("--block", "128x256x64", "--warps", "8", "--stages", "4")
        k = self.made("mm-f16", "1", None, None, *block)
        self.assertEqual(k["target"], "sm_80")
        self.assertLessEqual(k["shared"], 98304)
        self.assertEqual(k["num_stages"], 3)
        self.assertIn("stages of the 4 asked, for its shared memory", (self.root / "vf.log").read_text())

    def test_slower_products(self):
        # A bfloat16 product of a tile of 16 x 128 x 64 whose B is stored by rows (K x N), measured
        # slower for sm_80 on a V100 (SLOWER): compiled for 7.5, as Triton alone compiles it; the
        # same with B stored by columns (N x K): sm_80.
        block = ("--block", "16x128x64", "--stages", "3")
        k = self.made("mm-bf16", "1", None, None, *block)
        if k["triton"].startswith("3.3."):
            # (Triton 3.3 cannot compile a product of halves for 7.5: the rule is not applied.)
            self.assertEqual(k["target"], "sm_80")
            return
        self.assertEqual(k["target"], "sm_75")
        self.assertIn("compiled for sm_75: bfloat16 products of a skinny tile measured slower", (self.root / "vf.log").read_text())
        off = self.alone("mm-bf16", *block)
        if off is not None:
            self.assertEqual(k["cubin"], off["cubin"])
        self.assertEqual(self.made("mmt-bf16", "1", None, None, *block)["target"], "sm_80")

    def test_line_info_off(self):
        r = self.compile("mm-f16", "1", None, {"TRITON_DISABLE_LINE_INFO": "1"})
        if self.alone("mm-f16") is None:
            self.assertNotEqual(r.returncode, 0)
        else:
            self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1])["target"], "sm_75")


if __name__ == "__main__":
    unittest.main()
