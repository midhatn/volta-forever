"""CPU regression for kernels refused through the driver's linker, and in the cubins of the layer's
nvJitLink.

A kernel that cannot run on the GPU, in an input of cuLinkAddData, is refused when the program asks
for it in the image that cuLinkComplete made: as the program got it, and as a copy loaded after the
link is gone. A file given with cuLinkAddFile is adapted as data is. A kernel that a cubin's note
names (the layer's nvJitLink marks so a kernel that traps in the cubin it makes) is refused wherever
the cubin is loaded. A stand-in driver (link_mock_driver.c) links into a fixed cubin, whatever its
inputs, and has every kernel.

Run with python3 tests/proxy/link_refusals.py path/to/libcuda.so.1.
"""
import ctypes as c
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import refusal_note

PTX = (b".version 8.8\n.target sm_75\n.address_size 64\n"
       b".visible .entry bad()\n{\n barrier.cluster.arrive;\n ret;\n}\n"
       b".visible .entry good()\n{\n ret;\n}\n")
INPUT_CUBIN = 0
INPUT_PTX = 1
INPUT_FATBINARY = 2
INPUT_OBJECT = 3
INPUT_LIBRARY = 4
WRAPPER_MAGIC = 0x466243B1
CUBIN = Path(__file__).resolve().parents[1] / "kernels" / "k_basic.128.sm70.cubin"
# Machine code for sm_75 of which the translator refuses the function of warp_sum (tests/gpu/run.sh,
# cubin-sm75-damaged).
DAMAGED = Path(__file__).resolve().parents[1] / "kernels" / "k_basic.135.sm75.damaged.cubin"
WARP_SUM_REFUSED = ("kernel warp_sum cannot run on this GPU and is not given to the program: its machine code for "
                    "sm_75 is not translated (the instruction ffffffffffffffffffffffffffffffff at 0x0")


def lookup(api, module, name):
    function = c.c_void_p(0xDEADBEEF)
    status = api.cuModuleGetFunction(c.byref(function), module, name)
    return status, function.value


def load(api, image):
    module = c.c_void_p()
    assert api.cuModuleLoadData(c.byref(module), image) == 0
    return module


def fat_binary(cubin, sm=70):
    """`cubin` as the machine code for `sm`, stored as it is, of a fat binary (the header of the sm_70
    entry of a fat binary of the repository's, with this size and architecture)."""
    native = (Path(__file__).resolve().parents[2] / "crates/vf-fatbin/tests/data/elf70-elf75-ptx70-lz4.fatbin")
    header = bytearray(native.read_bytes()[16:16 + 64])
    stored = cubin + bytes(-len(cubin) % 8)
    struct.pack_into("<QI", header, 8, len(stored), 0)
    struct.pack_into("<Q", header, 0x28, struct.unpack_from("<Q", header, 0x28)[0] & ~0xB000)
    struct.pack_into("<Q", header, 0x38, 0)
    assert struct.unpack_from("<HI", header, 0)[0] == 2 and struct.unpack_from("<I", header, 0x1C)[0] == 70
    struct.pack_into("<I", header, 0x1C, sm)
    body = bytes(header) + stored
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(body)) + body


def entries(fat):
    """The kind (1 PTX, 2 machine code) and architecture of each entry of the fat binary `fat`."""
    found, at = [], 16
    while at < 16 + struct.unpack_from("<Q", fat, 8)[0]:
        kind, _, header, payload = struct.unpack_from("<HHIQ", fat, at)
        found.append((kind, struct.unpack_from("<I", fat, at + 0x1C)[0]))
        at += header + payload
    return found


def cubin_sm(cubin):
    """The architecture that the header of `cubin` names (in its flags, where its ABI version says)."""
    flags = struct.unpack_from("<I", cubin, 0x30)[0]
    return (flags >> 8) & 0xff if cubin[8] >= 8 else flags & 0xff


def given_as_data(backend):
    """The type and the bytes of the last input that the stand-in's linker was given as data."""
    data = c.c_void_p.in_dll(backend, "seen_data").value
    return c.c_int.in_dll(backend, "seen_type").value, c.string_at(data, c.c_size_t.in_dll(backend, "seen_size").value)


def given_as_file(backend):
    """The type and the path of the last input that the stand-in's linker was given as a file."""
    return c.c_int.in_dll(backend, "file_type").value, (c.c_char * 4096).in_dll(backend, "file_path").value


def counts(backend):
    """How many inputs the stand-in's linker was given as data, and as files."""
    return c.c_uint.in_dll(backend, "inputs").value, c.c_uint.in_dll(backend, "file_inputs").value


def said_since(start):
    """The lines that the layer wrote to its log (VF_LOG) after its first `start` bytes."""
    return Path(os.environ["VF_LOG"]).read_bytes()[start:].decode(errors="replace").splitlines()


class Wrapper(c.Structure):
    """What the CUDA runtime gives the driver for a fat binary: version 1, or 2 with the relocatable
    fat binaries to link if the first has nothing for the GPU (a list that ends with NULL)."""
    _fields_ = [("magic", c.c_uint), ("version", c.c_uint), ("data", c.c_void_p), ("extra", c.c_void_p)]


def wrappers(fat):
    """`fat` behind a wrapper of each version, with what they point to."""
    data = c.create_string_buffer(fat, len(fat))
    members = (c.c_void_p * 1)(None)
    made = []
    for version, extra in ((1, None), (2, c.addressof(members))):
        made.append(Wrapper(WRAPPER_MAGIC, version, c.addressof(data), extra))
    return made, (data, members)


def marked_everywhere(api, tmp, marked):
    """Each way of loading `marked` refuses the kernel `bad` that its note names, and gives `good`."""
    names = {name for name, _, _ in refusal_note.read(marked)}
    assert names == {"bad"}, names
    path = Path(tmp) / "marked.cubin"
    path.write_bytes(marked)
    fat = fat_binary(marked)
    (Path(tmp) / "marked.fatbin").write_bytes(fat)
    held, keep = wrappers(fat)
    wrong = []

    def module_ways():
        yield "cuModuleLoadData", lambda m: api.cuModuleLoadData(m, marked)
        yield "cuModuleLoadDataEx", lambda m: api.cuModuleLoadDataEx(m, marked, 0, None, None)
        yield "cuModuleLoad", lambda m: api.cuModuleLoad(m, os.fsencode(path))
        yield "cuModuleLoadData of a fat binary", lambda m: api.cuModuleLoadData(m, fat)
        yield "cuModuleLoad of a fat binary", lambda m: api.cuModuleLoad(m, os.fsencode(Path(tmp) / "marked.fatbin"))
        for wrapper in held:
            yield f"cuModuleLoadFatBinary of a wrapper {wrapper.version}", \
                lambda m, w=wrapper: api.cuModuleLoadFatBinary(m, c.byref(w))

    for way, call in module_ways():
        module = c.c_void_p()
        if call(c.byref(module)) != 0:
            wrong.append(f"{way}: not loaded")
            continue
        if lookup(api, module, b"bad") != (218, None) or lookup(api, module, b"good")[0] != 0:
            wrong.append(f"{way}: bad {lookup(api, module, b'bad')}, good {lookup(api, module, b'good')}")
        assert api.cuModuleUnload(module) == 0
    for way, call in (
        ("cuLibraryLoadData", lambda l: api.cuLibraryLoadData(l, marked, None, None, 0, None, None, 0)),
        ("cuLibraryLoadFromFile", lambda l: api.cuLibraryLoadFromFile(l, os.fsencode(path), None, None, 0, None, None, 0)),
        ("cuLibraryLoadData of a fat binary", lambda l: api.cuLibraryLoadData(l, fat, None, None, 0, None, None, 0)),
    ):
        library = c.c_void_p()
        if call(c.byref(library)) != 0:
            wrong.append(f"{way}: not loaded")
            continue
        kernel = c.c_void_p(0xDEADBEEF)
        bad = (api.cuLibraryGetKernel(c.byref(kernel), library, b"bad"), kernel.value)
        good = api.cuLibraryGetKernel(c.byref(kernel), library, b"good")
        if bad != (218, None) or good != 0:
            wrong.append(f"{way}: bad {bad}, good {good}")
        assert api.cuLibraryUnload(library) == 0
    # As an input of the driver's linker: the image it makes refuses the kernel.
    state, image, size = c.c_void_p(), c.c_void_p(), c.c_size_t()
    assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
    assert api.cuLinkAddData_v2(state, INPUT_CUBIN, marked, len(marked), b"marked.cubin", 0, None, None) == 0
    assert api.cuLinkComplete(state, c.byref(image), c.byref(size)) == 0
    if lookup(api, load(api, image), b"bad") != (218, None):
        wrong.append("a link of it: bad given")
    assert api.cuLinkDestroy(state) == 0
    del keep
    assert not wrong, "\n".join(wrong)


def check(proxy, mock):
    api, backend = c.CDLL(proxy), c.CDLL(mock)
    output = CUBIN.read_bytes()
    linked = c.create_string_buffer(output, len(output))
    c.c_void_p.in_dll(backend, "linked").value = c.addressof(linked)
    c.c_size_t.in_dll(backend, "linked_size").value = len(output)

    state = c.c_void_p()
    assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
    assert api.cuLinkAddData_v2(state, INPUT_PTX, PTX, len(PTX), b"bad.ptx", 0, None, None) == 0
    image, size = c.c_void_p(), c.c_size_t()
    assert api.cuLinkComplete(state, c.byref(image), c.byref(size)) == 0
    module = load(api, image)
    assert lookup(api, module, b"bad") == (218, None), lookup(api, module, b"bad")
    assert lookup(api, module, b"good")[0] == 0
    copy = c.create_string_buffer(c.string_at(image, size.value), size.value)
    assert api.cuLinkDestroy(state) == 0
    assert c.c_uint.in_dll(backend, "destroyed").value == 1
    assert lookup(api, load(api, copy), b"bad") == (218, None)
    # And in a fat binary of the program's own, as the machine code for this GPU.
    native = (Path(__file__).resolve().parents[2] / "crates/vf-fatbin/tests/data/elf70-elf75-ptx70-lz4.fatbin")
    header = bytearray(native.read_bytes()[16:16 + 64])
    stored = output + bytes(-len(output) % 8)
    struct.pack_into("<QI", header, 8, len(stored), 0)
    struct.pack_into("<Q", header, 0x28, struct.unpack_from("<Q", header, 0x28)[0] & ~0xB000)
    struct.pack_into("<Q", header, 0x38, 0)
    assert struct.unpack_from("<HI", header, 0)[0] == 2 and struct.unpack_from("<I", header, 0x1C)[0] == 70
    body = bytes(header) + stored
    wrapped = struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(body)) + body
    assert lookup(api, load(api, wrapped), b"bad") == (218, None)
    assert lookup(api, load(api, wrapped), b"good")[0] == 0
    # The same behind a wrapper of the CUDA runtime, of version 1 and of version 2 (until
    # 2026-11-23 version 2 gave bad), and as the relocatable fat binary of a version-2 wrapper whose
    # first fat binary has nothing for this GPU (an entry said to be for sm_90).
    held, keep = wrappers(wrapped)
    for wrapper in held:
        for loader in (api.cuModuleLoadData, api.cuModuleLoadFatBinary):
            module = c.c_void_p()
            assert loader(c.byref(module), c.byref(wrapper)) == 0
            assert lookup(api, module, b"bad") == (218, None), (wrapper.version, loader)
            assert lookup(api, module, b"good")[0] == 0
    elsewhere = bytearray(wrapped)
    struct.pack_into("<I", elsewhere, 16 + 0x1C, 90)
    first = c.create_string_buffer(bytes(elsewhere), len(elsewhere))
    members = (c.c_void_p * 2)(c.addressof(keep[0]), None)
    wrapper = Wrapper(WRAPPER_MAGIC, 2, c.addressof(first), c.addressof(members))
    module = c.c_void_p()
    assert api.cuModuleLoadFatBinary(c.byref(module), c.byref(wrapper)) == 0
    assert lookup(api, module, b"bad") == (218, None)
    assert lookup(api, module, b"good")[0] == 0

    # Other bytes are another image, and a link with nothing refused, given the handle of the link
    # that is gone, gives nothing to its image.
    other = bytearray(output)
    other[len(other) // 2] ^= 0xff
    assert lookup(api, load(api, bytes(other)), b"bad")[0] == 0
    good = PTX.replace(b"barrier.cluster.arrive;", b"ret;")
    other_linked = c.create_string_buffer(bytes(other), len(other))
    c.c_void_p.in_dll(backend, "linked").value = c.addressof(other_linked)
    assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
    assert api.cuLinkAddData_v2(state, INPUT_PTX, good, len(good), b"good.ptx", 0, None, None) == 0
    assert api.cuLinkComplete(state, c.byref(image), c.byref(size)) == 0
    assert lookup(api, load(api, image), b"bad")[0] == 0
    assert api.cuLinkDestroy(state) == 0
    assert c.c_uint.in_dll(backend, "inputs").value == 2

    # A link whose options name the architecture of its output (CU_JIT_TARGET) is given its inputs
    # for that one: adapted for this GPU when it is this GPU's, as they are when it is a later one.
    for target, expected in ((None, b".target sm_70"), (70, b".target sm_70"), (90, b".target sm_75"),
                             (0x10000 + 90, b".target sm_75")):
        options, values = (c.c_int * 1)(9), (c.c_void_p * 1)(target)
        count = 0 if target is None else 1
        assert api.cuLinkCreate_v2(count, options, values, c.byref(state)) == 0
        assert api.cuLinkAddData_v2(state, INPUT_PTX, PTX, len(PTX), b"bad.ptx", 0, None, None) == 0
        assert c.c_char_p(c.addressof(c.c_char.in_dll(backend, "seen_target"))).value == expected, target
        assert api.cuLinkDestroy(state) == 0

    # The image that a link gives without its size (sizeOut may be NULL) is known all the same.
    third = bytearray(output)
    third[len(third) // 3] ^= 0xff
    third_linked = c.create_string_buffer(bytes(third), len(third))
    c.c_void_p.in_dll(backend, "linked").value = c.addressof(third_linked)
    assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
    assert api.cuLinkAddData_v2(state, INPUT_PTX, PTX, len(PTX), b"bad.ptx", 0, None, None) == 0
    assert api.cuLinkComplete(state, c.byref(image), None) == 0
    assert lookup(api, load(api, image), b"bad") == (218, None)
    assert api.cuLinkDestroy(state) == 0

    # What another thread does with a handle that the driver gives again once it is free, while
    # the destruction of a link or the unload of a module returns, is not dropped with what was
    # known of the handle before: a link made then, and a module loaded then.
    fourth = bytearray(output)
    fourth[len(fourth) // 4] ^= 0xff
    fourth_linked = c.create_string_buffer(bytes(fourth), len(fourth))
    c.c_void_p.in_dll(backend, "linked").value = c.addressof(fourth_linked)
    assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
    assert api.cuLinkAddData_v2(state, INPUT_PTX, good, len(good), b"good.ptx", 0, None, None) == 0
    again = c.c_void_p()

    def link_meanwhile():
        assert api.cuLinkCreate_v2(0, None, None, c.byref(again)) == 0 and again.value == state.value
        assert api.cuLinkAddData_v2(again, INPUT_PTX, PTX, len(PTX), b"bad.ptx", 0, None, None) == 0

    meanwhile = c.CFUNCTYPE(None)(link_meanwhile)
    c.c_void_p.in_dll(backend, "while_freed").value = c.cast(meanwhile, c.c_void_p).value
    assert api.cuLinkDestroy(state) == 0
    c.c_void_p.in_dll(backend, "while_freed").value = None
    assert api.cuLinkComplete(again, c.byref(image), c.byref(size)) == 0
    assert lookup(api, load(api, image), b"bad") == (218, None)
    assert api.cuLinkDestroy(again) == 0
    first, second = load(api, PTX), c.c_void_p()

    def load_meanwhile():
        c.c_size_t.in_dll(backend, "next_module").value = first.value
        assert api.cuModuleLoadData(c.byref(second), PTX) == 0 and second.value == first.value

    meanwhile = c.CFUNCTYPE(None)(load_meanwhile)
    c.c_void_p.in_dll(backend, "while_freed").value = c.cast(meanwhile, c.c_void_p).value
    assert api.cuModuleUnload(first) == 0
    c.c_void_p.in_dll(backend, "while_freed").value = None
    assert lookup(api, second, b"bad") == (218, None)

    # A module or a library that the driver does not unload, and a link that it does not destroy,
    # are still the program's, and what is known of them stays. (Until 2026-11-23 the refused
    # kernel was given after an unload that failed, and the inputs of a link were forgotten.)
    marked = refusal_note.mark(output, [("bad", "bad", "an instruction without a Volta sequence")])
    library = c.c_void_p()
    assert api.cuLibraryLoadData(c.byref(library), marked, None, None, 0, None, None, 0) == 0
    c.c_uint.in_dll(backend, "unload_fails").value = 1
    assert api.cuModuleUnload(second) == 201
    assert lookup(api, second, b"bad") == (218, None)
    assert api.cuLibraryUnload(library) == 201
    kernel = c.c_void_p(0xDEADBEEF)
    assert (api.cuLibraryGetKernel(c.byref(kernel), library, b"bad"), kernel.value) == (218, None)
    c.c_uint.in_dll(backend, "unload_fails").value = 0
    assert api.cuModuleUnload(second) == 0 and api.cuLibraryUnload(library) == 0
    sixth = bytearray(output)
    sixth[len(sixth) // 6] ^= 0xff
    sixth_linked = c.create_string_buffer(bytes(sixth), len(sixth))
    c.c_void_p.in_dll(backend, "linked").value = c.addressof(sixth_linked)
    assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
    assert api.cuLinkAddData_v2(state, INPUT_PTX, PTX, len(PTX), b"bad.ptx", 0, None, None) == 0
    c.c_uint.in_dll(backend, "destroy_fails").value = 1
    assert api.cuLinkDestroy(state) == 400
    c.c_uint.in_dll(backend, "destroy_fails").value = 0
    assert api.cuLinkComplete(state, c.byref(image), c.byref(size)) == 0
    assert lookup(api, load(api, image), b"bad") == (218, None)
    assert api.cuLinkDestroy(state) == 0

    # A cubin that names its kernels that trap is refused them, loaded in any way. (Until
    # 2026-11-23 they were given to the program.)
    with tempfile.TemporaryDirectory() as tmp:
        fifth = bytearray(output)
        fifth[len(fifth) // 5] ^= 0xff
        fifth_linked = c.create_string_buffer(bytes(fifth), len(fifth))
        c.c_void_p.in_dll(backend, "linked").value = c.addressof(fifth_linked)
        marked_everywhere(api, tmp, refusal_note.mark(output, [("bad", "bad", "an instruction without a Volta sequence")]))

    # Through cuLinkAddFile and cuLinkAddFile_v2 on a PTX file: adapted for this GPU, with refused
    # kernels withheld after cuLinkComplete and forwarding via cuLinkAddData[_v2] (not raw cuLinkAddFile).
    with tempfile.TemporaryDirectory() as tmp:
        ptx_file = Path(tmp) / "bad_file.ptx"
        ptx_file.write_bytes(PTX)
        encoded = os.fsencode(ptx_file)
        for add_file in (api.cuLinkAddFile, api.cuLinkAddFile_v2):
            seventh = bytearray(output)
            seventh[len(seventh) // 7] ^= 0xff
            seventh_linked = c.create_string_buffer(bytes(seventh), len(seventh))
            c.c_void_p.in_dll(backend, "linked").value = c.addressof(seventh_linked)
            before_files = c.c_uint.in_dll(backend, "file_inputs").value
            assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
            assert add_file(state, INPUT_PTX, encoded, 0, None, None) == 0
            assert c.c_uint.in_dll(backend, "file_inputs").value == before_files, add_file.__name__
            assert c.c_char_p(c.addressof(c.c_char.in_dll(backend, "seen_target"))).value == b".target sm_70"
            assert api.cuLinkComplete(state, c.byref(image), c.byref(size)) == 0
            assert lookup(api, load(api, image), b"bad") == (218, None)
            assert lookup(api, load(api, image), b"good")[0] == 0
            assert api.cuLinkDestroy(state) == 0
        # CU_JIT_TARGET forwarding (sm_70 adapted via cuLinkAddData_v2; sm_90 passed as file untouched)
        for target, expected, delta_files in ((70, b".target sm_70", 0), (90, b".target sm_75", 1)):
            options, values = (c.c_int * 1)(9), (c.c_void_p * 1)(target)
            before_files = c.c_uint.in_dll(backend, "file_inputs").value
            assert api.cuLinkCreate_v2(1, options, values, c.byref(state)) == 0
            assert api.cuLinkAddFile_v2(state, INPUT_PTX, encoded, 0, None, None) == 0
            assert c.c_uint.in_dll(backend, "file_inputs").value == before_files + delta_files, target
            assert c.c_char_p(c.addressof(c.c_char.in_dll(backend, "seen_target"))).value == expected, target
            assert api.cuLinkDestroy(state) == 0
        # A file that cannot be read, and a null path, go to the driver's call as they are, and the
        # program gets its error (the stand-in's: 301 and 1).
        for path, error in ((os.fsencode(Path(tmp) / "missing.ptx"), 301), (None, 1)):
            for add_file in (api.cuLinkAddFile, api.cuLinkAddFile_v2):
                before = counts(backend)
                assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
                assert add_file(state, INPUT_PTX, path, 0, None, None) == error, (path, add_file.__name__)
                assert counts(backend) == (before[0], before[1] + 1), (path, add_file.__name__)
                assert given_as_file(backend) == (INPUT_PTX, path or b"")
                assert api.cuLinkDestroy(state) == 0
        # What the layer made of a file, refused by the driver: the driver is given the file, and the
        # program gets what that call returns.
        c.c_uint.in_dll(backend, "data_fails").value = 1
        start = Path(os.environ["VF_LOG"]).stat().st_size
        for add_file in (api.cuLinkAddFile, api.cuLinkAddFile_v2):
            before = counts(backend)
            assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
            assert add_file(state, INPUT_PTX, encoded, 0, None, None) == 0, add_file.__name__
            assert counts(backend) == (before[0] + 1, before[1] + 1), add_file.__name__
            assert given_as_file(backend) == (INPUT_PTX, encoded)
            assert api.cuLinkDestroy(state) == 0
        c.c_uint.in_dll(backend, "data_fails").value = 0
        said = [line for line in said_since(start)
                if "cuLinkAddFile: the driver refused the adapted input (error 218); adding the file" in line]
        assert len(said) == 2, said

    # A cubin and a fat binary of sm_75 machine code, in files, through both: the linker is given,
    # as data, what the layer makes of the same bytes given as data (machine code for sm_70), and in
    # the image that the link makes the kernel that the translator refuses is refused, with the line
    # of a refused kernel (once: the first time), and another is given.
    damaged = DAMAGED.read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        links = 0
        for kind, contents in ((INPUT_CUBIN, damaged), (INPUT_FATBINARY, fat_binary(damaged, 75))):
            path = os.fsencode(Path(tmp) / f"damaged.{kind}")
            Path(os.fsdecode(path)).write_bytes(contents)
            assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
            assert api.cuLinkAddData_v2(state, kind, contents, len(contents), None, 0, None, None) == 0
            adapted = given_as_data(backend)
            assert api.cuLinkDestroy(state) == 0
            assert adapted[0] == kind and adapted[1] != contents, kind
            if kind == INPUT_CUBIN:
                assert cubin_sm(adapted[1]) == 70
            else:
                assert (2, 70) in entries(adapted[1]) and (2, 75) not in entries(adapted[1]), entries(adapted[1])
            for add_file in (api.cuLinkAddFile, api.cuLinkAddFile_v2):
                links += 1
                made = bytearray(output)
                made[len(made) // 8 + links] ^= 0xff
                made_linked = c.create_string_buffer(bytes(made), len(made))
                c.c_void_p.in_dll(backend, "linked").value = c.addressof(made_linked)
                before, start = counts(backend), Path(os.environ["VF_LOG"]).stat().st_size
                assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
                assert add_file(state, kind, path, 0, None, None) == 0
                assert counts(backend) == (before[0] + 1, before[1]), (kind, add_file.__name__)
                assert given_as_data(backend) == adapted, (kind, add_file.__name__)
                assert api.cuLinkComplete(state, c.byref(image), c.byref(size)) == 0
                module = load(api, image)
                assert lookup(api, module, b"warp_sum") == (218, None), (kind, add_file.__name__)
                assert lookup(api, module, b"saxpy")[0] == 0
                said = [line for line in said_since(start) if WARP_SUM_REFUSED in line]
                assert len(said) == (links == 1), said
                assert api.cuLinkDestroy(state) == 0
        # A file of another type (a host object, a library of them) goes to the driver's call as it
        # is, whatever its bytes: here the cubin's, which the layer translates as a cubin.
        path = os.fsencode(Path(tmp) / f"damaged.{INPUT_CUBIN}")
        for kind in (INPUT_OBJECT, INPUT_LIBRARY):
            for add_file in (api.cuLinkAddFile, api.cuLinkAddFile_v2):
                before = counts(backend)
                assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
                assert add_file(state, kind, path, 0, None, None) == 0
                assert counts(backend) == (before[0], before[1] + 1), (kind, add_file.__name__)
                assert given_as_file(backend) == (kind, path)
                assert api.cuLinkDestroy(state) == 0

    # A fat binary given with its size is read within it, whatever its header says: one cut short,
    # just before memory that cannot be read, is given to the linker as it is.
    fat = (Path(__file__).resolve().parents[2] / "crates/vf-fatbin/tests/data/elf75-ptx75-zstd.fatbin").read_bytes()
    page = mmap.PAGESIZE
    guard = -(-len(fat) // page) * page
    pages = mmap.mmap(-1, page + guard)
    start = c.addressof(c.c_char.from_buffer(pages))
    libc = c.CDLL(None)
    libc.mprotect.argtypes = (c.c_void_p, c.c_size_t, c.c_int)
    assert libc.mprotect(start + page, guard, 0) == 0
    cut = 200
    pages[page - cut:page] = fat[:cut]
    state = c.c_void_p()
    assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
    inputs = c.c_uint.in_dll(backend, "inputs").value
    assert api.cuLinkAddData_v2(state, INPUT_FATBINARY, c.c_void_p(start + page - cut), cut, None, 0, None, None) == 0
    assert c.c_uint.in_dll(backend, "inputs").value == inputs + 1
    # And one of no bytes, at memory that cannot be read: nothing of it is read (until 2026-11-23 a
    # size of 0 made the layer read a header there, and the process ended).
    assert api.cuLinkAddData_v2(state, INPUT_FATBINARY, c.c_void_p(start + page), 0, None, 0, None, None) == 0
    assert c.c_uint.in_dll(backend, "inputs").value == inputs + 2
    assert api.cuLinkDestroy(state) == 0


if __name__ == "__main__":
    if len(sys.argv) == 3:
        check(sys.argv[1], sys.argv[2])
    else:
        with tempfile.TemporaryDirectory() as tmp:
            mock = str(Path(tmp) / "link.so")
            subprocess.run(["cc", "-shared", "-fPIC", "-o", mock, Path(__file__).with_name("link_mock_driver.c")],
                           check=True)
            # The layer's lines in a file of their own, which the test reads (and shows on a failure).
            env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0", VF_LOG=str(Path(tmp) / "layer.log"))
            status = subprocess.run([sys.executable, __file__, sys.argv[1], mock], env=env).returncode
            if status != 0:
                log = Path(env["VF_LOG"])
                sys.exit(f"FAIL kernels refused through the linker (status {status}); the layer said:\n"
                         + (log.read_text(errors="replace") if log.exists() else "nothing"))
            print("PASS kernels refused through the linker")
