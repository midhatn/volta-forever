"""volta-forever: the helper of VF_TRITON_TC, which compiles kernels for sm_80 for one process.

Started by that process (volta_forever_triton_client says what a helper is, how long it lives and
where it writes) as `PYTHON -I -S .../volta_forever_triton_helper.py --helper SOCKET SLOT WAITING
PARENT ROOT`: the descriptors of its socket, of the lock of its slot and of the lock "waiting"
(-1 for a helper without a slot), the pid of that process and the directory that holds the
program's Triton. The rules of what is compiled for sm_80 are in volta_forever_triton. No program
reads this file.
"""

import os
import sys


def _load(name, filename):
    """The module of this file's directory named so, read without writing its bytecode."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    module = sys.modules.get(name)
    if module is not None and getattr(module, "__file__", None) == path:
        return module
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    written, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = written
    return module


_core = _load("volta_forever_triton", "volta_forever_triton.py")
_client = _load("volta_forever_triton_client", "volta_forever_triton_client.py")
slower, VOLTA_SHARED, keeps_ptx = _core.slower, _core.VOLTA_SHARED, _client.keeps_ptx

# A helper that has had nothing to do for GRACE seconds gives its slot up to a process that waits
# for one (the lock "waiting" is held), which it looks at every POLL seconds.
GRACE = 0.5
POLL = 0.1
# An answer is taken by the program in SEND seconds, or the helper ends.
SEND = 60.0
# The directory of this helper's temporary files: its path and a descriptor of it.
_work = {"path": None, "fd": None}


_adapt_worker = {"proc": None, "key": None, "unsupported": set()}


def _close_worker_proc(proc):
    for s in (proc.stdin, proc.stdout, proc.stderr):
        if s is not None:
            try:
                s.close()
            except OSError:
                pass


def _stop_adapt_worker():
    proc = _adapt_worker.get("proc")
    _adapt_worker["proc"] = None
    _adapt_worker["key"] = None
    if proc is None:
        return
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=0.5)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
            proc.wait(timeout=0.5)
        except Exception:  # noqa: BLE001
            pass
    _close_worker_proc(proc)


def _read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("short read from vf-translate adapt-worker")
        buf.extend(chunk)
    return bytes(buf)


def _ensure_adapt_worker(tool, env):
    import subprocess
    key = (tool, tuple(sorted(env.items())))
    if tool in _adapt_worker["unsupported"]:
        return None
    proc = _adapt_worker["proc"]
    if proc is not None and _adapt_worker["key"] == key and proc.poll() is None:
        return proc
    _stop_adapt_worker()
    try:
        proc = subprocess.Popen(
            [tool, "adapt-worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            close_fds=True,
        )
        hdr = proc.stdout.read(4)
    except (OSError, subprocess.SubprocessError):
        return None
    if hdr != b"VFAW":
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=0.5)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
                proc.wait(timeout=0.5)
            except Exception:  # noqa: BLE001
                pass
        _close_worker_proc(proc)
        _adapt_worker["unsupported"].add(tool)
        return None
    _adapt_worker["proc"] = proc
    _adapt_worker["key"] = key
    return proc


def _layer_takes_oneshot(tool, env, cubin):
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix="vf-triton-") as d:
        path = os.path.join(d, "kernel.cubin")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _client._O_CLOEXEC, 0o600)
        with open(fd, "wb") as f:
            f.write(cubin)
        try:
            r = subprocess.run([tool, "adapt", path, os.path.join(d, "adapted")], env=env, capture_output=True,
                               timeout=300, check=False)
        except (OSError, subprocess.SubprocessError) as e:
            return "the layer's tool did not run (%s)" % e, True
    out = r.stdout.decode("utf-8", "replace")
    if r.returncode != 0:
        return "the layer's tool failed: %s" % (out + r.stderr.decode("utf-8", "replace")).strip()[:300], True
    if "PTX for sm_80 retargeted" not in out:
        return "the layer would not retarget its PTX: %s" % out.strip()[:300], False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("refused") or s.startswith("trap:"):
            return "the layer would not run it all: %s" % s[:300], False
    return None, False


def layer_takes(cubin):
    """(None, False) when the layer's tool retargets the PTX that the cubin keeps, with no kernel
    refused and no function made to trap, else (why not, whether that may be otherwise another
    time: the tool did not run, or failed). The tool reads the switches of the layer (VF_*) as
    the layer does."""
    import struct
    tool = _core._tool()
    if tool is None:
        return "the layer's tool vf-translate was not found", True
    env = {k: v for k, v in os.environ.items() if k not in ("VF_LOG", "VF_LOG_LEVEL", "LD_AUDIT", "LD_PRELOAD")}
    proc = _ensure_adapt_worker(tool, env)
    if proc is None:
        return _layer_takes_oneshot(tool, env, cubin)
    try:
        proc.stdin.write(struct.pack("<Q", len(cubin)))
        proc.stdin.write(cubin)
        proc.stdin.flush()
        status_hdr = _read_exact(proc.stdout, 5)
        status = status_hdr[0]
        msg_len = struct.unpack("<I", status_hdr[1:5])[0]
        msg = _read_exact(proc.stdout, msg_len).decode("utf-8", "replace") if msg_len else ""
    except (OSError, EOFError, struct.error):
        _stop_adapt_worker()
        return _layer_takes_oneshot(tool, env, cubin)
    if status == 0:
        return None, False
    if status == 1:
        return "the layer would not run it all: %s" % msg[:300], False
    if status == 2:
        return "the layer's tool failed: %s" % msg[:300], True
    return "the layer would not retarget its PTX: %s" % msg[:300], False


def _parse(text, context, suffix):
    """A module of Triton's IR from its text, as Triton reads a .ttir or .ttgir file."""
    import tempfile
    from triton._C.libtriton import ir
    fd, path = tempfile.mkstemp(prefix="vf-triton-", suffix="." + suffix)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        module = ir.parse_mlir_module(path, context)
    finally:
        os.unlink(path)
    module.context = context
    return module


def compile80(backend, ttir, context, metadata, options):
    """The sm_80 kernel of a TTIR: (what each stage makes, None, False), or (None, why not,
    whether that may be otherwise another time). Done on a copy of the module, with a copy of the
    metadata: what fails leaves both as they were. The metadata of the kernel says how many
    stages it was compiled with."""
    import dataclasses
    why = None
    for stages in range(max(1, options.num_stages), 0, -1):
        md = dict(metadata)
        opt = dataclasses.replace(options, arch="sm80", num_stages=stages)
        ttgir = backend.make_ttgir(_parse(ttir, context, "ttir"), md, opt, 80)
        ttgir_text = str(ttgir)
        rule = slower(ttir, ttgir_text, max(1, options.num_stages))
        if rule:
            return None, rule, False
        llir = backend.make_llir(ttgir, md, opt, 80)
        shared = md.get("shared")
        if not isinstance(shared, int):
            return None, "the size of its shared memory is not known (%r)" % (shared,), False
        if shared > VOLTA_SHARED:
            why = "%d bytes of shared memory with %d stages" % (shared, stages)
            continue
        ptx = backend.make_ptx(llir, md, opt, 80)
        cubin = backend.make_cubin(ptx, md, opt, 80)
        if not keeps_ptx(cubin):
            return None, "its cubin does not keep its PTX", False
        refused, transient = layer_takes(cubin)
        if refused:
            return None, refused, transient
        md["num_stages"] = stages
        return dict(ttgir=ttgir_text, llir=llir, ptx=ptx, cubin=cubin, metadata=md, stages=stages), None, False
    return None, why, False


def _from_outside(e):
    """Why an error of a compilation is not the kernel's, or None: ptxas (which Triton runs, and
    whose failure it raises as an error of its own: the CalledProcessError is in the chain, and
    Triton's text says "error code N") ended by a signal of another than a crash (SIGKILL as from
    the out-of-memory killer, SIGTERM, SIGINT, SIGHUP), or a full disk or quota."""
    import errno
    import re
    import subprocess
    seen = 0
    while e is not None and seen < 8:
        seen += 1
        codes = []
        if isinstance(e, subprocess.CalledProcessError):
            codes.append(e.returncode)
        codes += [int(c) for c in re.findall(r"error code (-?\d+)", str(e))]
        for code in codes:
            # (A status below 0 is a signal's; above 128, a shell's for a signal, when ptxas is a
            # script that runs it: only those of an end from outside are taken so.)
            sig = -code if code < 0 else code - 128 if code - 128 in (1, 2, 9, 15) else 0
            if sig > 0 and sig not in _client.CRASH_SIGNALS:
                return "ptxas was ended from outside (signal %d)" % sig
        if isinstance(e, OSError) and e.errno in (errno.ENOSPC, errno.EDQUOT):
            return "no space left on its device (%s)" % e.strerror
        if re.search(r"No space left on device|Disk quota exceeded", str(e)):
            return "no space left on its device"
        e = e.__cause__ or e.__context__
    return None


def _checked_open(file, mode="r", *args, **kwargs):
    """open() for Triton's NVIDIA compiler in the helper (in its module's globals, in place of the
    builtin): a file that it reads (the cubin that ptxas wrote, ptxas's log) is opened through the
    descriptor of the helper's directory when it is there, without following a link, and must be
    a regular file (not a FIFO, which would hold it) of CUBIN_MAX bytes at most; OSError otherwise.
    What it writes is as the builtin does."""
    import builtins
    import errno
    import stat
    if not isinstance(file, (str, bytes, os.PathLike)) or "r" not in mode or "+" in mode:
        return builtins.open(file, mode, *args, **kwargs)
    path = os.fsdecode(file)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | _client._O_CLOEXEC
    work = _work["path"]
    if work and path.startswith(work + os.sep):
        fd = os.open(path[len(work) + 1:], flags, dir_fd=_work["fd"])
    else:
        fd = os.open(path, flags)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(errno.EINVAL, "%s is not a regular file" % path)
        if st.st_size > _client.CUBIN_MAX:
            raise OSError(errno.EFBIG, "%s has %d bytes" % (path, st.st_size))
        os.set_blocking(fd, True)
        return builtins.open(fd, mode, *args, **kwargs)
    except BaseException:
        os.close(fd)
        raise


def _request_problem(request, opt_cls, triton_dir, default_libs):
    """Why a request is not one that a helper compiles, or None. Each of its values is checked,
    not only its names: a TTIR (text), CUDA's target for 7.5 (what a process asks), the options of
    Triton's CUDAOptions (opt_cls) with values of their kinds (no IR given in place of a stage, no
    instrumentation, ptxas's options of PTXAS_FLAGS alone, libraries of Triton's directory alone,
    one CTA, the numbers of warps, stages and registers in their ranges, other texts short and
    without a path), a metadata of JSON."""
    import dataclasses
    import re
    if not isinstance(request.get("ttir"), str) or not isinstance(request.get("options"), dict) \
            or not isinstance(request.get("metadata"), dict) or not isinstance(request.get("target"), dict):
        return "a request that is not a compilation"
    if request["target"] != {"backend": "cuda", "arch": 75, "warp_size": 32}:
        return "a target that is not CUDA's for 7.5"
    options = request["options"]
    names = {f.name for f in dataclasses.fields(opt_cls)}
    if not set(options) <= names:
        return "options that Triton does not have: %s" % sorted(set(options) - names)[:5]
    why = _core.options_problem(options, triton_dir, default_libs)
    if why:
        return why.replace("its ", "", 1)

    def number(name, low, high):
        v = options.get(name)
        return v is None or (isinstance(v, int) and not isinstance(v, bool) and low <= v <= high)

    if options.get("arch", "sm75") != "sm75" or options.get("num_ctas", 1) != 1 or options.get("warp_size", 32) != 32:
        return "options of another target than 7.5 with one CTA"
    if not (number("num_warps", 1, 64) and number("num_stages", 0, 16) and number("maxnreg", 1, 255)):
        return "numbers of warps, stages or registers out of their ranges"

    def plain(v, depth=0):
        if v is None or isinstance(v, (bool, int, float)):
            return True
        if isinstance(v, str):
            return re.fullmatch(r"[A-Za-z0-9_.+-]{0,64}", v) is not None
        if isinstance(v, list) and depth < 2 and len(v) <= 64:
            return all(plain(x, depth + 1) for x in v)
        return False

    for name, v in options.items():
        if name not in ("ptx_options", "extern_libs") and not plain(v):
            return "a value of %s that is not one of Triton's" % name
    return None


def _tuples(value):
    if isinstance(value, list):
        return tuple(_tuples(v) for v in value)
    return value


def _compile(request, known):
    """compile80 of a request, validated: the answer."""
    from triton._C.libtriton import ir
    if "cls" not in known:
        import triton
        from triton.backends import backends
        cls = backends["nvidia"].compiler
        cubin_globals = getattr(getattr(cls, "make_cubin", None), "__globals__", None)
        if isinstance(cubin_globals, dict):
            cubin_globals["open"] = _checked_open   # (this helper's alone)
        GPUTarget = getattr(sys.modules.get(cls.__module__), "GPUTarget", None)
        if GPUTarget is None:
            from triton.backends.compiler import GPUTarget
        opt_cls = type(cls(GPUTarget("cuda", 75, 32)).parse_options({}))
        known.update(cls=cls, target=GPUTarget, options=opt_cls, dir=os.path.dirname(triton.__file__),
                     libs=_core._default_libs(opt_cls))
    why = _request_problem(request, known["options"], known["dir"], known["libs"])
    if why:
        return {"kind": "invalid", "why": why}
    backend = known["cls"](known["target"](**request["target"]))
    options = known["options"](**{k: _tuples(v) for k, v in request["options"].items()})
    metadata = dict(request["metadata"])
    if isinstance(metadata.get("target"), dict):
        metadata["target"] = known["target"](**metadata["target"])
    context = ir.context() if hasattr(ir, "context") else None
    if context is not None:
        ir.load_dialects(context)
        backend.load_dialects(context)
    result, why, transient = compile80(backend, request["ttir"], context, metadata, options)
    if result is None:
        return {"kind": "transient" if transient else "decision", "why": why}
    return {"kind": "ok", "result": result}


def _answer(request, known, err):
    """The answer to one message of the program (never an exception), with the request's digest
    and what the compiler wrote meanwhile."""
    try:
        digest = _client._digest(request) if request is not None else None
    except (TypeError, ValueError, RecursionError):
        digest = None
    if digest is None or request.get("digest") != digest:
        return {"kind": "invalid", "why": "a request whose digest is not that of what it asks"}
    os.lseek(err, 0, os.SEEK_SET)
    os.ftruncate(err, 0)
    try:
        answer = _compile(request, known)
    except (OSError, MemoryError) as e:
        answer = {"kind": "transient", "why": "%s: %s" % (type(e).__name__, str(e).strip()[:300])}
    except Exception as e:  # noqa: BLE001
        first = str(e).strip().splitlines()[0][:300] if str(e).strip() else ""
        outside = _from_outside(e)
        if outside:   # (not the kernel's: not kept, compiled again another time)
            answer = {"kind": "transient", "why": "its sm_80 compilation failed: %s" % outside}
        else:
            answer = {"kind": "error", "why": "an error of its sm_80 compilation: %s: %s" % (type(e).__name__, first)}
    for f in (sys.stdout, sys.stderr):
        try:
            f.flush()
        except (OSError, ValueError):
            pass
    os.lseek(err, 0, os.SEEK_SET)
    answer.update(digest=digest, stderr=os.read(err, 65536).decode("utf-8", "replace"))
    try:
        _client._message(answer)
    except (TypeError, ValueError):
        answer = {"kind": "error", "why": "an answer that could not be sent", "digest": digest}
    return answer


def _others_wait(fd):
    """Whether a process of the user waits for a slot of this host (it holds the lock "waiting"
    shared, which this helper then cannot take alone)."""
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return False
    fcntl.flock(fd, fcntl.LOCK_UN)
    return False


def _work_dir():
    """The directory of this helper's temporary files: made with mode 0700 (mkdtemp) in the
    temporary directory (the program's TMPDIR), where Triton's own compilation writes the same
    files. The helper removes it when it ends, the program's waiter after a crash; one of a helper
    killed with its program is left to the system's cleaning of temporary files, as Triton's own
    files are. No other directory is looked at or removed."""
    import tempfile
    return tempfile.mkdtemp(prefix="volta-forever-triton-")


def _end_children():
    """What this helper still runs (a ptxas left by a compilation that failed) killed, and its
    children reaped (a second at most), when it ends otherwise than killed: none of them is then
    left to its program, or to a child subreaper, to reap. (The program's waiter kills what is left
    of the helper's group after it ends, whatever ended it.)"""
    import glob
    import signal
    import time
    _stop_adapt_worker()
    kids = set()
    for path in glob.glob("/proc/self/task/*/children"):
        try:
            with open(path) as f:
                kids.update(int(p) for p in f.read().split())
        except (OSError, ValueError):
            continue
    for pid in kids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    end = time.monotonic() + 1.0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except OSError:   # (ECHILD: none left)
            return
        if pid == 0:
            if not kids or time.monotonic() > end:
                return
            time.sleep(0.02)
        kids.discard(pid)


def _serve(sock, waiting, err):
    """Answer the program's requests, one at a time, until its socket ends, the helper has had
    nothing to do for VF_TRITON_TC_IDLE seconds, or for GRACE seconds while a process waits for a
    slot, or an answer is not taken. A compilation that has not ended a minute after
    VF_TRITON_TC_TIMEOUT (twice that time-out when it is shorter) ends the helper (SIGALRM): the
    program kills it at the time-out, unless it has ended meanwhile in a way that the helper could
    not see (its PR_SET_PDEATHSIG is handled only between two instructions of Python, and LLVM's or
    MLIR's code may not end)."""
    import select
    import signal
    import time
    idle = _client._seconds("VF_TRITON_TC_IDLE", _client.IDLE)
    timeout = _client._seconds("VF_TRITON_TC_TIMEOUT", _client.TIMEOUT)
    limit = int(timeout + min(timeout, 60.0)) + 1
    inbox = _client._Inbox(sock, _client.REQUEST_MAX)
    poll = select.poll()
    poll.register(sock.fileno(), select.POLLIN)
    known = {}
    last = time.monotonic()
    while True:
        quiet = time.monotonic() - last
        if quiet >= idle or (quiet >= GRACE and waiting >= 0 and _others_wait(waiting)):
            return
        if not poll.poll(min(idle - quiet, POLL) * 1000.0):
            continue
        last = time.monotonic()
        try:
            data = inbox.read()
        except (EOFError, ValueError):
            return
        if data is None:
            continue
        signal.alarm(limit)
        answer = _answer(_client._decode(data), known, err)
        signal.alarm(0)
        try:
            _client._send(sock, _client._message(answer), SEND)
        except OSError:
            return
        last = time.monotonic()


def _main(sock_fd, slot_fd, waiting_fd, parent, root):
    import shutil
    import signal
    import socket
    import tempfile
    for fd in (sock_fd, slot_fd, waiting_fd):
        if fd >= 0:
            os.set_inheritable(fd, False)   # (not to ptxas, which Triton runs with close_fds=False)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def parent_gone(*_):
        # PR_SET_PDEATHSIG: sent when the thread that started the helper ends, and when its process
        # does. The end of the process alone (the helper is then another's child) ends the helper
        # and its group (ptxas).
        if os.getppid() != parent:
            if _work["path"]:
                shutil.rmtree(_work["path"], ignore_errors=True)
            os.killpg(0, signal.SIGKILL)

    signal.signal(signal.SIGUSR2, parent_gone)
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGUSR2), 0, 0, 0)
    except Exception:  # noqa: BLE001 (its socket still ends with the program)
        pass
    if os.getppid() != parent:
        return
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=sock_fd)
    sock.setblocking(False)
    try:
        work = _work_dir()
    except OSError as e:
        _client._send(sock, _client._message({"kind": "failed", "why": "its temporary directory could not be made "
                                                                       "(%s)" % (e.strerror or e)}), SEND)
        return
    try:
        _work.update(path=work, fd=os.open(work, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _client._O_CLOEXEC))
        os.environ["TMPDIR"] = tempfile.tempdir = work
        err = os.open(os.path.join(work, "stderr"), os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _client._O_CLOEXEC,
                      0o600)
        os.dup2(err, 1)
        os.dup2(err, 2)
        # (An import of Triton that does not end, as from a network disk that hangs, ends the helper
        # after VF_TRITON_TC_TIMEOUT seconds: SIGALRM, whose default ends a process whatever it runs.)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(int(_client._seconds("VF_TRITON_TC_TIMEOUT", _client.TIMEOUT)))
        try:
            # (Triton from the directory that holds the program's, after the standard library: the
            # helper has nothing else of the program's sys.path.)
            if root not in sys.path:
                sys.path.append(root)
            import triton  # noqa: F401
            from triton.backends import backends  # noqa: F401
            from triton._C.libtriton import ir  # noqa: F401
            hello = {"kind": "ready", "identity": _client._identity(), "work": work, "pid": os.getpid()}
        except BaseException as e:  # noqa: BLE001
            hello = {"kind": "failed", "why": "%s: %s" % (type(e).__name__, str(e).strip()[:300])}
        finally:
            signal.alarm(0)
        _client._send(sock, _client._message(hello), SEND)
        if hello["kind"] == "ready":
            _serve(sock, waiting_fd, err)
    finally:
        _end_children()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__" and sys.argv[1:2] == ["--helper"] and len(sys.argv) == 7:
    _main(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
