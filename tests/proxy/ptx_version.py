"""The two architectures of a kernel's code that a program can ask about (cuFuncGetAttribute,
cuKernelGetAttribute: CU_FUNC_ATTRIBUTE_PTX_VERSION, 5, and CU_FUNC_ATTRIBUTE_BINARY_VERSION, 6), by
every way that the code of PTX for a later GPU reaches the V100, against a stand-in for the driver
(guard_mock_driver.c) that answers as NVIDIA's does: the architecture of the device for the machine
code, and for the PTX the latest `.target` of the PTX it was given that the device reaches.

The layer retargets PTX for sm_75 to sm_70, which the driver then says the code was compiled from:
CUB's and Thrust's algorithms, which choose by that answer among the architectures they were built
for, answered "invalid device function" (until 2026-11-24). A Turing or later GPU that compiles PTX
for sm_75 says 7.5; through the layer, a function or a kernel whose code it retargeted from PTX for
sm_NN is said to be compiled from sm_NN, at most the capability that its caller is told (and its
machine code, as before, for the capability told). What the driver says stands for a caller with
code of its own for the V100, for PTX that the layer did not retarget, and on a later card.

Ways: a module loaded from memory and from a file, a library loaded from memory (deferred, its
kernels compiled alone, or whole) and from a file, its kernel asked about as a kernel and as a
function, the kernel's function, the function of the library's module in the context, a kernel
asked for after that module was made; set up from the layer's cache at the second start. Callers:
a program whose code is PTX for sm_75 only (as a CUDA 13 build for 75-virtual), the CUDA 13 runtime,
and a library with code for sm_70 (told the truth); told by default, with VF_CAPABILITY=8.0 (and
VF_CAPABILITY_ALL=1) and 7.0, under the launcher and not.

Run with python3 tests/proxy/ptx_version.py path/to/libcuda.so.1.
"""
import ctypes as c
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "crates/vf-fatbin/tests/data"

# A normal (non-tail) call makes this object the return address that the layer sees.
CLIENT = """
int load(int (*f)(void **, const void *), void **m, const void *image) { return f(m, image); }
int load_file(int (*f)(void **, const char *), void **m, const char *path) { return f(m, path); }
int library(int (*f)(void **, const void *, int *, void **, unsigned, int *, void **, unsigned), void **l,
            const void *image) {
    return f(l, image, 0, 0, 0, 0, 0, 0);
}
int library_file(int (*f)(void **, const char *, int *, void **, unsigned, int *, void **, unsigned), void **l,
                 const char *path) {
    return f(l, path, 0, 0, 0, 0, 0, 0);
}
int attribute(int (*f)(int *, int, void *), int *v, int a, void *h) { return f(v, a, h); }
int kernel_attribute(int (*f)(int *, int, void *, int), int *v, int a, void *k, int d) { return f(v, a, k, d); }
"""


def text(sm):
    """PTX for `sm` with two kernels, a and b, which a deferred library compiles alone."""
    return (f".version 8.8\n.target sm_{sm}\n.address_size 64\n"
            ".visible .entry a()\n{\n ret;\n}\n.visible .entry b()\n{\n ret;\n}\n").encode()


def entries(image):
    """The (header, stored payload) entries of a fat binary."""
    size, offset, found = struct.unpack_from("<Q", image, 8)[0], 16, []
    while offset < 16 + size:
        header_size = struct.unpack_from("<I", image, offset + 4)[0]
        stored_size = struct.unpack_from("<Q", image, offset + 8)[0]
        found.append((image[offset:offset + header_size], image[offset + header_size:offset + header_size + stored_size]))
        offset += header_size + stored_size
    return found


def fat_binary(targets):
    """A fat binary of PTX entries for `targets`, stored as they are (as nvcc makes for
    CMAKE_CUDA_ARCHITECTURES=75-virtual, or 75-virtual;80-virtual)."""
    template = entries((DATA / "elf75-ptx75-zstd.fatbin").read_bytes())[1][0]
    body = b""
    for sm in targets:
        payload = text(sm)
        stored = payload + bytes(8 - len(payload) % 8)
        header = bytearray(template)
        struct.pack_into("<QI", header, 8, len(stored), 0)
        struct.pack_into("<HH", header, 0x18, 8, 8)
        struct.pack_into("<I", header, 0x1C, sm)
        struct.pack_into("<Q", header, 0x28, struct.unpack_from("<Q", header, 0x28)[0] & ~0xB000)
        struct.pack_into("<Q", header, 0x38, 0)
        body += bytes(header) + stored
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(body)) + body


# The images, with the architectures of their PTX; and the clients, with what is in their file.
IMAGES = {
    "text75": text(75) + b"\0",
    "fat75": fat_binary([75]),
    "fat75-80": fat_binary([75, 80]),
    # The control: PTX for the V100, which the layer does not change.
    "text70": text(70) + b"\0",
}
CLIENTS = ("ptx75.so", "native.so", "runtimes/libcudart.so.13")


def build(directory):
    subprocess.run(["cc", "-shared", "-fPIC", "-Wl,-Bsymbolic", "-o", str(directory / "guard.so"),
                    str(Path(__file__).with_name("guard_mock_driver.c"))], check=True)
    (directory / "runtimes").mkdir()
    plain = directory / "plain.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-O0", "-Wl,-Bsymbolic", "-x", "c", "-", "-o", str(plain)],
                   input=CLIENT, text=True, check=True)
    subprocess.run(["cc", "-shared", "-fPIC", "-O0", "-Wl,-Bsymbolic", "-Wl,-soname,libcudart.so.13", "-x", "c", "-",
                    "-o", str(directory / "runtimes/libcudart.so.13")], input=CLIENT, text=True, check=True)
    (directory / "ptx75.fatbin").write_bytes(fat_binary([75]))
    for name, fixture in (("ptx75.so", directory / "ptx75.fatbin"), ("native.so", DATA / "elf70-elf75-ptx70-lz4.fatbin")):
        subprocess.run(["objcopy", "--add-section", f".nv_fatbin={fixture}", str(plain), str(directory / name)],
                       check=True)
    for name, image in IMAGES.items():
        (directory / f"{name}.image").write_bytes(image.rstrip(b"\0") if name.startswith("text") else image)


def asked():
    """What VF_CAPABILITY asks for, as an SM number (None: not set)."""
    value = os.environ.get("VF_CAPABILITY")
    return int(value.replace(".", "")) if value else None


def told(client):
    """What `client` is told, as an SM number: None for the truth. The program whose code is PTX for
    sm_75 is told no later generation than that (it is no CUDA runtime, which VF_CAPABILITY_ALL=1
    would tell what was asked); the runtime, which no code limits here, what was asked."""
    given = asked()
    if client == "native.so":
        return None
    if client == "ptx75.so":
        return given if given and given // 10 <= 7 else 75
    return given or 75


def given_sm(image, client):
    """The architecture of the PTX that the layer gives the V100 for `image` loaded by `client`: the
    PTX for what the images of the client are adapted for (what VF_CAPABILITY asks, but where the
    client's code limits what it is told), where the image has some, else the nearest above."""
    targets = {"text75": [75], "fat75": [75], "fat75-80": [75, 80], "text70": [70]}[image]
    given = asked()
    at = (told(client) if client == "ptx75.so" else given) or 70
    return max([sm for sm in targets if sm <= at], default=min(targets))


def expected(image, client, device, how, lazy):
    """The PTX and binary versions that `client` is answered about code of `image` on `device`, the
    way `how` (a module, else a library: deferred when `lazy`). On the later card (device 1), what
    the driver says of the code that runs there: a module loaded there is the program's, and so are
    the entries of a fat binary that a library keeps beside the V100's; a library deferred by the
    layer, or loaded from PTX text, runs the layer's PTX there too, and is said to be compiled from
    the program's, whoever asks."""
    targets = {"text75": [75], "fat75": [75], "fat75-80": [75, 80], "text70": [70]}[image]
    if device == 1:
        if how == "function of a module" or image == "text70" or (not lazy and image.startswith("fat")):
            return max(targets), 90
        return given_sm(image, client), 90
    caller = told(client)
    if caller is None or caller <= 70:
        return 70, 70
    sm = given_sm(image, client)
    return (min(sm, caller) if sm > 70 else 70), caller


def case(proxy, directory):
    api = c.CDLL(proxy)
    lazy = os.environ.get("VF_LAZY") != "0"
    context, later = c.c_void_p(), c.c_void_p()
    assert api.cuInit(0) == 0
    assert api.cuCtxCreate_v2(c.byref(later), 0, 1) == 0
    assert api.cuCtxCreate_v2(c.byref(context), 0, 0) == 0
    wrong = []

    def ask(client, handle, how, image, device=0):
        want = expected(image, client.label, device, how, lazy)
        got = []
        for attribute in (5, 6):
            value = c.c_int(-1)
            if how == "kernel":
                status = client.kernel_attribute(api.cuKernelGetAttribute, c.byref(value), attribute, handle, device)
            else:
                status = client.attribute(api.cuFuncGetAttribute, c.byref(value), attribute, handle)
            assert status == 0, (client.label, image, how, attribute, status)
            got.append(value.value)
        if tuple(got) != want:
            wrong.append(f"{client.label}, {image}, {how} on device {device}: PTX {got[0]}, binary {got[1]}; "
                         f"expected PTX {want[0]}, binary {want[1]}")

    def function(module, name):
        handle = c.c_void_p()
        assert api.cuModuleGetFunction(c.byref(handle), module, name) == 0, name
        return handle

    def kernel(library, name):
        handle = c.c_void_p()
        assert api.cuLibraryGetKernel(c.byref(handle), library, name) == 0, name
        return handle

    for label in CLIENTS:
        client = c.CDLL(str(directory / label))
        client.label = label.split("/")[-1]
        for image in IMAGES:
            path = directory / f"{image}.image"
            data = c.create_string_buffer(IMAGES[image], len(IMAGES[image]))
            assert api.cuCtxSetCurrent(context) == 0
            # Modules, from memory and from a file.
            for load in (lambda m: client.load(api.cuModuleLoadData, c.byref(m), data),
                         lambda m: client.load_file(api.cuModuleLoad, c.byref(m), str(path).encode())):
                module = c.c_void_p()
                assert load(module) == 0, image
                ask(client, function(module, b"a"), "function of a module", image)
                assert api.cuModuleUnload(module) == 0
            # A library from memory: deferred or loaded whole, as VF_LAZY says.
            library = c.c_void_p()
            assert client.library(api.cuLibraryLoadData, c.byref(library), data) == 0, image
            first = kernel(library, b"a")
            ask(client, first, "kernel", image)
            ask(client, first, "kernel given as a function", image)
            own = c.c_void_p()
            assert api.cuKernelGetFunction(c.byref(own), first) == 0
            ask(client, own, "function of a kernel", image)
            module = c.c_void_p()
            assert api.cuLibraryGetModule(c.byref(module), library) == 0
            ask(client, function(module, b"b"), "function of the library's module", image)
            ask(client, kernel(library, b"b"), "kernel asked for after the module", image)
            # The same kernel on the later card, and its function in a context there (of a library
            # loaded whole from PTX text, the module of that context is not known here: not asked).
            ask(client, first, "kernel", image, device=1)
            if lazy or not image.startswith("text"):
                assert api.cuCtxSetCurrent(later) == 0
                there = c.c_void_p()
                assert api.cuKernelGetFunction(c.byref(there), first) == 0
                ask(client, there, "function of a kernel", image, device=1)
                assert api.cuCtxSetCurrent(context) == 0
            assert api.cuLibraryUnload(library) == 0
            # A library from a file, loaded whole.
            library = c.c_void_p()
            assert client.library_file(api.cuLibraryLoadFromFile, c.byref(library), str(path).encode()) == 0, image
            ask(client, kernel(library, b"a"), "kernel of a library from a file", image)
            assert api.cuLibraryUnload(library) == 0
            # A module loaded on the later card is the program's, as the driver says.
            assert api.cuCtxSetCurrent(later) == 0
            module = c.c_void_p()
            assert client.load(api.cuModuleLoadData, c.byref(module), data) == 0
            ask(client, function(module, b"a"), "function of a module", image, device=1)
            assert api.cuModuleUnload(module) == 0
    if wrong:
        sys.exit("\n".join(wrong))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[3] == "--case":
        case(sys.argv[1], Path(sys.argv[2]))
        sys.exit(0)
    if len(sys.argv) != 2:
        sys.exit("usage: python3 tests/proxy/ptx_version.py LIBCUDA (the layer's, as dist/lib/libcuda.so.1)")
    proxy = str(Path(sys.argv[1]).resolve(strict=True))
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        build(directory)
        base = {k: v for k, v in os.environ.items() if not k.startswith("VF_") and k != "VOLTA_FOREVER"}
        base.update(VF_REAL_LIBCUDA=str(directory / "guard.so"), VF_LOG_LEVEL="1")
        runs = []
        for told_settings in ({}, {"VF_CAPABILITY": "8.0"}, {"VF_CAPABILITY": "8.0", "VF_CAPABILITY_ALL": "1"},
                              {"VF_CAPABILITY": "7.0"}):
            for launched in ({}, {"VOLTA_FOREVER": "test"}):
                for lazy in ("0", "1"):
                    runs.append(({**told_settings, **launched, "VF_LAZY": lazy, "VF_PROXY_CACHE": "0"}, None))
        # The second start takes the deferred libraries from the layer's cache.
        for told_settings in ({}, {"VF_CAPABILITY": "8.0"}):
            cache = directory / f"cache-{len(runs)}"
            for start in ("first start", "from the cache"):
                runs.append(({**told_settings, "VF_LAZY": "1", "VF_PROXY_CACHE": str(cache), "VF_TIMING": "1"}, start))
        for settings, start in runs:
            done = subprocess.run([sys.executable, __file__, proxy, str(directory), "--case"], env={**base, **settings},
                                  capture_output=True, text=True)
            label = " ".join(f"{k}={v}" for k, v in settings.items() if k != "VF_TIMING") + (f" ({start})" if start else "")
            ok = done.returncode == 0
            if start == "from the cache" and "cache hit" not in done.stderr:
                ok = False
                print("    the libraries were not set up from the cache")
            if start == "from the cache" and (
                "retargeted from the cache" not in done.stderr or "PTX retargeted in" in done.stderr
            ):
                ok = False
                print("    retargeted PTX modules were not loaded from the cache")
            print(f"{'ok  ' if ok else 'FAIL'}  {label}")
            if not ok:
                print("    " + (done.stderr.strip() or done.stdout.strip()).replace("\n", "\n    "))
            failed |= not ok
    print("FAIL PTX and binary versions" if failed else
          "PASS PTX and binary versions: retargeted code is said to be compiled from the program's PTX, by every way")
    sys.exit(1 if failed else 0)
