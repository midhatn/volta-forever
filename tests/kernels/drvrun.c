// drvrun: load a GPU module (PTX text, cubin ELF or fat binary) through the CUDA driver API and run
// the k_basic.cu kernels found in it, checking their results against the host.
//
// usage: drvrun [-d ORDINAL] [-r REPEAT] [-q] MODULE_FILE
//
// With DRVRUN_LINK=cuLinkAddFile or cuLinkAddFile_v2, MODULE_FILE goes to the driver's linker by its
// path, through that entry point (as PTX, a cubin or a fat binary, by its first bytes), and the image
// that the link makes is loaded.
//
// Only libcuda.so.1 is needed (dlopen): no CUDA runtime, so the test isolates the driver's behaviour.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef unsigned long long CUdeviceptr;
enum {  // CUjit_option
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_LOG_VERBOSE = 12,
};

static void* lib;
#define SYM(ret, name, ...) static ret (*name)(__VA_ARGS__)
SYM(CUresult, cuInit, unsigned);
SYM(CUresult, cuDriverGetVersion, int*);
SYM(CUresult, cuDeviceGet, CUdevice*, int);
SYM(CUresult, cuDeviceGetName, char*, int, CUdevice);
SYM(CUresult, cuDeviceComputeCapability, int*, int*, CUdevice);
SYM(CUresult, cuCtxCreate_v2, CUcontext*, unsigned, CUdevice);
SYM(CUresult, cuCtxDestroy_v2, CUcontext);
SYM(CUresult, cuCtxSynchronize, void);
SYM(CUresult, cuModuleLoadDataEx, CUmodule*, const void*, unsigned, int*, void**);
SYM(CUresult, cuModuleGetFunction, CUfunction*, CUmodule, const char*);
SYM(CUresult, cuFuncGetAttribute, int*, int, CUfunction);
SYM(CUresult, cuMemAlloc_v2, CUdeviceptr*, size_t);
SYM(CUresult, cuMemFree_v2, CUdeviceptr);
SYM(CUresult, cuMemcpyHtoD_v2, CUdeviceptr, const void*, size_t);
SYM(CUresult, cuMemcpyDtoH_v2, void*, CUdeviceptr, size_t);
SYM(CUresult, cuMemsetD8_v2, CUdeviceptr, unsigned char, size_t);
SYM(CUresult, cuLaunchKernel, CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
    void*, void**, void**);
SYM(CUresult, cuGetErrorName, CUresult, const char**);

static void* sym(const char* name) {
    void* p = dlsym(lib, name);
    if (!p) { fprintf(stderr, "missing symbol %s\n", name); exit(2); }
    return p;
}
static const char* ename(CUresult r) {
    const char* s = "?";
    cuGetErrorName(r, &s);
    return s ? s : "?";
}
#define CK(call) do { CUresult r_ = (call); if (r_) { printf("FAIL %s -> %d %s\n", #call, r_, ename(r_)); exit(1); } } while (0)

static int quiet;
static int failures;
// DRVRUN_KERNELS=name,name restricts the kernels that run (default: all found in the module)
static int want(const char* kernel) {
    const char* list = getenv("DRVRUN_KERNELS");
    if (!list) return 1;
    size_t n = strlen(kernel);
    for (const char* p = list; (p = strstr(p, kernel)); p += n)
        if ((p == list || p[-1] == ',') && (p[n] == 0 || p[n] == ',')) return 1;
    return 0;
}
static void verdict(const char* kernel, long bad, long n, double maxerr) {
    if (bad) failures++;
    if (!quiet || bad) printf("  %-10s %s  (%ld/%ld wrong, max err %.3g)\n", kernel, bad ? "WRONG" : "ok", bad, n, maxerr);
}

// IEEE half <-> float for small exact values (host side reference)
static uint16_t f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, exp = (x >> 23) & 0xff, man = x & 0x7fffff;
    if (exp == 0) return (uint16_t)sign;
    int e = (int)exp - 127 + 15;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (e << 10) | (man >> 13));
}

#define N (1 << 20)
#define BLOCK 256
#define GRID (N / BLOCK)

static CUdeviceptr dalloc(size_t bytes) { CUdeviceptr p; CK(cuMemAlloc_v2(&p, bytes)); return p; }
static void launch(CUfunction f, void** args) {
    CK(cuLaunchKernel(f, GRID, 1, 1, BLOCK, 1, 1, 0, NULL, args, NULL));
    CUresult r = cuCtxSynchronize();
    if (r) { printf("FAIL launch/sync -> %d %s\n", r, ename(r)); exit(1); }
}

static void run_all(CUmodule mod) {
    CUfunction f;
    int n = N;
    float* x = malloc(N * sizeof(float));
    float* y = malloc(N * sizeof(float));
    CUdeviceptr dx = dalloc(N * 8), dy = dalloc(N * 8), dz = dalloc(N * 8);

    if (want("saxpy") && !cuModuleGetFunction(&f, mod, "saxpy")) {
        // The architecture the driver says the kernel's machine code is for (6: CU_FUNC_ATTRIBUTE_BINARY_VERSION).
        int version = 0;
        if (!cuFuncGetAttribute(&version, 6, f)) printf("machine code of saxpy: for %d.%d\n", version / 10, version % 10);
        float a = 2.0f;
        for (int i = 0; i < N; i++) { x[i] = i * 0.5f; y[i] = 1.0f; }
        CK(cuMemcpyHtoD_v2(dx, x, N * 4)); CK(cuMemcpyHtoD_v2(dy, y, N * 4));
        void* args[] = {&a, &dx, &dy, &n};
        launch(f, args);
        CK(cuMemcpyDtoH_v2(y, dy, N * 4));
        long bad = 0;
        for (int i = 0; i < N; i++) bad += y[i] != (float)i + 1.0f;
        verdict("saxpy", bad, N, 0);
    }
    if (want("mathk") && !cuModuleGetFunction(&f, mod, "mathk")) {
        for (int i = 0; i < N; i++) x[i] = (i - N / 2) * (8.0f / N);
        CK(cuMemcpyHtoD_v2(dx, x, N * 4));
        void* args[] = {&dx, &dy, &n};
        launch(f, args);
        CK(cuMemcpyDtoH_v2(y, dy, N * 4));
        long bad = 0; double maxerr = 0;
        for (int i = 0; i < N; i++) {
            double v = x[i];
            double ref = sin(v) * exp(-v * v) + sqrt(fabs(v)) + pow(fabs(v) + 1.0, 1.5);
            double err = fabs(y[i] - ref) / (fabs(ref) + 1e-6);
            if (err > maxerr) maxerr = err;
            bad += !(err < 1e-5);
        }
        verdict("mathk", bad, N, maxerr);
    }
    if (want("block_sum") && !cuModuleGetFunction(&f, mod, "block_sum")) {
        for (int i = 0; i < N; i++) x[i] = (float)(i % 13);
        CK(cuMemcpyHtoD_v2(dx, x, N * 4));
        void* args[] = {&dx, &dy, &n};
        launch(f, args);
        CK(cuMemcpyDtoH_v2(y, dy, GRID * 4));
        long bad = 0;
        for (int b = 0; b < GRID; b++) {
            float ref = 0;
            for (int t = 0; t < BLOCK; t++) ref += x[b * BLOCK + t];
            bad += y[b] != ref;
        }
        verdict("block_sum", bad, GRID, 0);
    }
    if (want("warp_sum") && !cuModuleGetFunction(&f, mod, "warp_sum")) {
        double ref = 0;
        for (int i = 0; i < N; i++) { x[i] = (float)(i % 7); ref += x[i]; }
        CK(cuMemcpyHtoD_v2(dx, x, N * 4)); CK(cuMemsetD8_v2(dy, 0, 4));
        void* args[] = {&dx, &dy, &n};
        launch(f, args);
        CK(cuMemcpyDtoH_v2(y, dy, 4));
        verdict("warp_sum", y[0] != (float)ref, 1, fabs(y[0] - ref));
    }
    if (want("histo") && !cuModuleGetFunction(&f, mod, "histo")) {
        unsigned* u = (unsigned*)x; unsigned bins[64], ref[64] = {0};
        for (int i = 0; i < N; i++) { u[i] = (unsigned)i * 2654435761u >> 7; ref[u[i] & 63]++; }
        CK(cuMemcpyHtoD_v2(dx, u, N * 4)); CK(cuMemsetD8_v2(dy, 0, sizeof bins));
        void* args[] = {&dx, &dy, &n};
        launch(f, args);
        CK(cuMemcpyDtoH_v2(bins, dy, sizeof bins));
        long bad = 0;
        for (int b = 0; b < 64; b++) bad += bins[b] != ref[b];
        verdict("histo", bad, 64, 0);
    }
    if (want("half_dot") && !cuModuleGetFunction(&f, mod, "half_dot")) {
        uint16_t* ha = (uint16_t*)x; uint16_t* hb = (uint16_t*)y; float* out = malloc(N * 4);
        for (int i = 0; i < N; i++) { ha[i] = f2h((float)(i % 31)); hb[i] = f2h((float)(i % 17)); }
        CK(cuMemcpyHtoD_v2(dx, ha, N * 2)); CK(cuMemcpyHtoD_v2(dy, hb, N * 2));
        void* args[] = {&dx, &dy, &dz, &n};
        launch(f, args);
        CK(cuMemcpyDtoH_v2(out, dz, N * 4));
        long bad = 0;
        for (int i = 0; i < N; i++) {
            float ref = (float)((i % 31) * (i % 17)) + (float)__builtin_popcount(i) + (float)__builtin_clz(i + 1);
            bad += out[i] != ref;
        }
        verdict("half_dot", bad, N, 0);
        free(out);
    }
    if (want("dbl") && !cuModuleGetFunction(&f, mod, "dbl")) {
        double* d = malloc(N * 8); double* o = malloc(N * 8);
        for (int i = 0; i < N; i++) d[i] = (i - N / 2) * (4.0 / N);
        CK(cuMemcpyHtoD_v2(dx, d, N * 8));
        void* args[] = {&dx, &dy, &n};
        launch(f, args);
        CK(cuMemcpyDtoH_v2(o, dy, N * 8));
        long bad = 0; double maxerr = 0;
        for (int i = 0; i < N; i++) {
            double ref = fma(d[i], 1.000001, 1.0 / (d[i] * d[i] + 1.0)) + (double)(((long long)i * 2654435761LL) % 1000);
            double err = fabs(o[i] - ref) / (fabs(ref) + 1e-9);
            if (err > maxerr) maxerr = err;
            bad += !(err < 1e-12);
        }
        verdict("dbl", bad, N, maxerr);
        free(d); free(o);
    }
    if (want("tanhk") && !cuModuleGetFunction(&f, mod, "tanhk")) {
        // inputs from -12 to 12, denser near 0 where a formula for tanh loses precision
        for (int i = 0; i < N; i++) { double u = (i - N / 2) * (2.0 / N); x[i] = (float)(12.0 * u * u * u); }
        CK(cuMemcpyHtoD_v2(dx, x, N * 4));
        void* args[] = {&dx, &dy, &n};
        launch(f, args);
        CK(cuMemcpyDtoH_v2(y, dy, N * 4));
        long bad = 0; double maxabs = 0, maxrel = 0;
        for (int i = 0; i < N; i++) {
            double ref = tanh((double)x[i]), err = fabs(y[i] - ref), rel = ref != 0 ? err / fabs(ref) : err;
            if (err > maxabs) maxabs = err;
            if (rel > maxrel) maxrel = rel;
            bad += !(err <= 1e-6 && rel <= 1e-4);  // NaN counts as wrong
        }
        verdict("tanhk", bad, N, maxabs);
        if (!quiet) printf("             max relative error %.3g\n", maxrel);
    }
    if (want("memory_check") && !cuModuleGetFunction(&f, mod, "memory_check")) {
        // Place each half between sentinels and check the entire destination after one warp.
        enum { lanes = 32, words = 2 * lanes + 1 };
        uint16_t input[words], result[words], expected[words];
        for (int i = 0; i < words; i++)
            input[i] = result[i] = expected[i] = 0x3555;
        float multiplier = 2.0f;
        uint32_t step = 2 * sizeof(uint16_t);
        for (int i = 0; i < lanes; i++) {
            float x = (float)(i - 16) * 0.25f;
            input[2 * i + 1] = f2h(x);
            expected[2 * i + 1] = f2h((x + 0.75f) * multiplier);
        }
        CK(cuMemcpyHtoD_v2(dx, input, sizeof input));
        CK(cuMemcpyHtoD_v2(dy, result, sizeof result));
        CUdeviceptr src = dx + sizeof(uint16_t), dst = dy + sizeof(uint16_t);
        void* args[] = {&src, &dst, &step, &multiplier};
        CK(cuLaunchKernel(f, 1, 1, 1, lanes, 1, 1, 0, NULL, args, NULL));
        CK(cuCtxSynchronize());
        CK(cuMemcpyDtoH_v2(result, dy, sizeof result));
        long bad = 0;
        for (int i = 0; i < words; i++) bad += result[i] != expected[i];
        verdict("memory_check", bad, words, 0);
        CUresult refused = cuModuleGetFunction(&f, mod, "unsupported_atomic");
        printf("lookup unsupported_atomic -> %d %s\n", refused, ename(refused));
    }
    if (want("guarded_copy") && !cuModuleGetFunction(&f, mod, "guarded_copy")) {
        uint32_t input[136], result[12];
        for (size_t i = 0; i < 136; i++) input[i] = 0x12348000u + i;
        CK(cuMemcpyHtoD_v2(dx, input, sizeof input));
        CUdeviceptr src = dx + 16;
        long bad = 0;
        for (uint32_t active = 0; active <= 1; active++) {
            CK(cuMemsetD8_v2(dy, 0xa5, sizeof result));
            void* args[] = {&src, &dy, &active};
            CK(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL));
            CK(cuCtxSynchronize());
            CK(cuMemcpyDtoH_v2(result, dy, sizeof result));
            // Offsets are relative to src. The unguarded copy reads dx+528,
            // while the guarded copy reads dx. Both write only dy+16..31.
            for (int i = 0; i < 12; i++) {
                uint32_t expected = 0xa5a5a5a5u;
                if (i >= 4 && i < 8) expected = input[(active ? 0 : 132) + i - 4];
                bad += result[i] != expected;
            }
        }
        verdict("guarded_copy", bad, 24, 0);
    }
    if (want("checked_add") && !cuModuleGetFunction(&f, mod, "checked_add")) {
        uint32_t inputs[] = {0, 123, 0xfffffff0u, 0x7fffffffu, 0x80000000u};
        CUdeviceptr destination = dy + sizeof(uint32_t);
        long bad = 0;
        for (size_t i = 0; i < sizeof inputs / sizeof inputs[0]; i++) {
            uint32_t result[3];
            CK(cuMemsetD8_v2(dy, 0xa5, sizeof result));
            void* args[] = {&destination, &inputs[i]};
            CK(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL));
            CK(cuCtxSynchronize());
            CK(cuMemcpyDtoH_v2(result, dy, sizeof result));
            bad += result[0] != 0xa5a5a5a5u;
            bad += result[1] != inputs[i] + 42u;
            bad += result[2] != 0xa5a5a5a5u;
        }
        verdict("checked_add", bad, 3 * (sizeof inputs / sizeof inputs[0]), 0);
    }
    if (want("histogram") && !cuModuleGetFunction(&f, mod, "histogram")) {
        enum { lanes = 32, words = 2 * lanes + 8, trials = 32 };
        uint32_t keys[lanes], result[words], expected[words];
        CUdeviceptr counters = dy + 4 * sizeof(uint32_t);
        long bad = 0;
        for (unsigned t = 0; t < trials; t++) {
            unsigned active = (t == trials-1) ? lanes : t;
            for (unsigned i = 0; i < words; i++) result[i] = expected[i] = 0xdecafbad;
            for (unsigned i = 0; i < lanes; i++) {
                keys[i] = (i * 7 + t) % (1 + t);
                result[4 + i] = expected[4 + i] = 0xfffffffcu + i;
                expected[4 + lanes + i] = result[4 + i];
            }
            for (unsigned i = 0; i < active; i++) expected[4 + keys[i]]++;
            CK(cuMemcpyHtoD_v2(dx, keys, sizeof keys));
            CK(cuMemcpyHtoD_v2(dy, result, sizeof result));
            void *args[] = {&counters, &dx, &active};
            CK(cuLaunchKernel(f, 1, 1, 1, lanes, 1, 1, 0, NULL, args, NULL));
            CK(cuCtxSynchronize());
            CK(cuMemcpyDtoH_v2(result, dy, sizeof result));
            for (unsigned i = 0; i < words; i++) bad += result[i] != expected[i];
        }
        verdict("histogram", bad, trials * words, 0);
    }
    if (want("warp_scan") && !cuModuleGetFunction(&f, mod, "warp_scan")) {
        uint32_t result[66];
        CUdeviceptr output = dx + sizeof(uint32_t);
        long bad = 0;
        for (unsigned n = 0; n <= 64; n += 32) {
            CK(cuMemsetD8_v2(dx, 0xa5, sizeof result));
            void *args[] = {&output, &n};
            CK(cuLaunchKernel(f, 1, 1, 1, 64, 1, 1, 0, NULL, args, NULL));
            CK(cuCtxSynchronize());
            CK(cuMemcpyDtoH_v2(result, dx, sizeof result));
            bad += result[0] != 0xa5a5a5a5u;
            bad += result[65] != 0xa5a5a5a5u;
            for (unsigned i = 0; i < 64; i++) {
                uint32_t expected = 0u - i;
                if (i < n) {
                    expected = 0;
                    for (unsigned j = 0; j <= i % 32 && j < 8; j++) expected += i - j;
                }
                bad += result[i + 1] != expected;
            }
        }
        verdict("warp_scan", bad, 3 * 66, 0);
    }
    CK(cuMemFree_v2(dx)); CK(cuMemFree_v2(dy)); CK(cuMemFree_v2(dz));
    free(x); free(y);
}

int main(int argc, char** argv) {
    int ordinal = 0, repeat = 1, a = 1;
    for (; a < argc && argv[a][0] == '-'; a++) {
        if (!strcmp(argv[a], "-d")) ordinal = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-r")) repeat = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-q")) quiet = 1;
    }
    if (a >= argc) { fprintf(stderr, "usage: drvrun [-d ORDINAL] [-r REPEAT] [-q] MODULE_FILE\n"); return 2; }
    const char* path = argv[a];
    FILE* fp = fopen(path, "rb");
    if (!fp) { perror(path); return 2; }
    fseek(fp, 0, SEEK_END); long size = ftell(fp); fseek(fp, 0, SEEK_SET);
    char* image = calloc(size + 1, 1);
    if (fread(image, 1, size, fp) != (size_t)size) { perror("read"); return 2; }
    fclose(fp);

    const char* libname = getenv("DRVRUN_LIBCUDA") ? getenv("DRVRUN_LIBCUDA") : "libcuda.so.1";
    lib = dlopen(libname, RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { fprintf(stderr, "dlopen %s: %s\n", libname, dlerror()); return 2; }
#define LOAD(name) name = sym(#name)
    LOAD(cuInit); LOAD(cuDriverGetVersion); LOAD(cuDeviceGet); LOAD(cuDeviceGetName); LOAD(cuDeviceComputeCapability);
    LOAD(cuCtxCreate_v2); LOAD(cuCtxDestroy_v2); LOAD(cuCtxSynchronize); LOAD(cuModuleLoadDataEx);
    LOAD(cuModuleGetFunction); LOAD(cuFuncGetAttribute); LOAD(cuMemAlloc_v2); LOAD(cuMemFree_v2); LOAD(cuMemcpyHtoD_v2); LOAD(cuMemcpyDtoH_v2);
    LOAD(cuMemsetD8_v2); LOAD(cuLaunchKernel); LOAD(cuGetErrorName);

    CUdevice dev; CUcontext ctx; char name[128]; int major, minor, drv;
    CK(cuInit(0));
    CK(cuDriverGetVersion(&drv));
    CK(cuDeviceGet(&dev, ordinal));
    CK(cuDeviceGetName(name, sizeof name, dev));
    CK(cuDeviceComputeCapability(&major, &minor, dev));
    if (!quiet) printf("device %d: %s, sm_%d%d, driver API %d\n", ordinal, name, major, minor, drv);
    CK(cuCtxCreate_v2(&ctx, 0, dev));

    static char info[1 << 16], err[1 << 16];
    int opts[] = {CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER,
                  CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_LOG_VERBOSE};
    void* vals[] = {info, (void*)(uintptr_t)sizeof info, err, (void*)(uintptr_t)sizeof err, (void*)(uintptr_t)1};
    CUmodule mod;
    CUresult r;
    const char* link = getenv("DRVRUN_LINK");
    if (link) {
        CUresult (*create)(unsigned, int*, void**, void**) = sym("cuLinkCreate_v2");
        CUresult (*add_file)(void*, int, const char*, unsigned, int*, void**) = sym(link);
        CUresult (*complete)(void*, void**, size_t*) = sym("cuLinkComplete");
        CUresult (*destroy)(void*) = sym("cuLinkDestroy");
        uint32_t magic = 0;
        memcpy(&magic, image, size < 4 ? (size_t)size : 4);
        // CU_JIT_INPUT_CUBIN, CU_JIT_INPUT_FATBINARY, else CU_JIT_INPUT_PTX
        int type = size >= 4 && !memcmp(image, "\177ELF", 4) ? 0 : magic == 0xBA55ED50u ? 2 : 1;
        void *state = NULL, *linked = NULL;
        size_t linked_size = 0;
        r = create(5, opts, vals, &state);
        if (!r) {
            r = add_file(state, type, path, 0, NULL, NULL);
            printf("link: %s of input type %d -> %d %s\n", link, type, r, r ? ename(r) : "CUDA_SUCCESS");
        }
        if (!r) {
            r = complete(state, &linked, &linked_size);
            printf("link: complete -> %d %s\n", r, r ? ename(r) : "CUDA_SUCCESS");
        }
        if (!r) r = cuModuleLoadDataEx(&mod, linked, 0, NULL, NULL);
        if (state) destroy(state);
    } else {
        r = cuModuleLoadDataEx(&mod, image, 5, opts, vals);
    }
    printf("load %s -> %d %s\n", path, r, r ? ename(r) : "CUDA_SUCCESS");
    if (err[0]) printf("  jit error log: %s\n", err);
    if (info[0] && !quiet) printf("  jit info log: %.600s\n", info);
    if (r) { cuCtxDestroy_v2(ctx); return 1; }
    for (int i = 0; i < repeat; i++) run_all(mod);
    if (repeat > 1) printf("  %d passes, %d failed checks\n", repeat, failures);
    CK(cuCtxDestroy_v2(ctx));
    return failures ? 1 : 0;
}
