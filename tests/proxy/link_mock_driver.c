// Driver for link_refusals.py and nccl_remedy.py: a linker whose output is the image the test puts
// in `linked`, with one handle for every link state (as an allocator may give a freed one again),
// and modules that have every kernel they are asked for. What another thread may do while the
// driver destroys a link or unloads a module, once the handle is free (`while_freed`), is done
// there; the next module loaded can be given a handle chosen by the test (`next_module`). An unload
// fails with 201 while `unload_fails` is set, a link destroy with 400 while `destroy_fails` is.
// With MOCK_LOAD_SLEEP=SECONDS in the environment, a module load and a link's completion say when
// they begin and end on standard error and take that long between, as the driver's compilation of a
// large image does (reported_traps.py: what the layer says comes before it).
#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const void *linked;
size_t linked_size;
unsigned inputs, links, destroyed;
void (*while_freed)(void);
uintptr_t next_module;
unsigned unload_fails, destroy_fails;

int cuLinkCreate_v2(unsigned count, int *options, void **values, void **state) {
    ++links;
    *state = (void *)(uintptr_t)0x1000;
    return 0;
}
/* The .target line of the last PTX input. */
char seen_target[32];
/* The last input given as data: its type, its size and a copy of its bytes. While `data_fails` is
   set, it is refused with 218. */
int seen_type = -1;
size_t seen_size;
unsigned char *seen_data;
unsigned data_fails;
int cuLinkAddData_v2(void *state, int type, void *data, size_t size, const char *name, unsigned count,
                     int *options, void **values) {
    ++inputs;
    seen_type = type;
    seen_size = size;
    free(seen_data);
    seen_data = malloc(size + 1);
    if (size) memcpy(seen_data, data, size);
    const char *line = type == 1 ? memmem(data, size, ".target", 7) : 0;
    memset(seen_target, 0, sizeof seen_target);
    for (size_t n = 0; line && n < sizeof seen_target - 1 && line[n] != '\n'; ++n) seen_target[n] = line[n];
    return data_fails ? 218 : 0;
}
int cuLinkAddData(void *state, int type, void *data, size_t size, const char *name, unsigned count,
                  int *options, void **values) {
    return cuLinkAddData_v2(state, type, data, size, name, count, options, values);
}
/* The last input given as a file: its type and its path. A null path is refused with
   CUDA_ERROR_INVALID_VALUE (1), one that cannot be opened with CUDA_ERROR_FILE_NOT_FOUND (301). */
unsigned file_inputs;
int file_type = -1;
char file_path[4096];
int cuLinkAddFile_v2(void *state, int type, const char *path, unsigned count, int *options, void **values) {
    ++file_inputs;
    file_type = type;
    memset(seen_target, 0, sizeof seen_target);
    memset(file_path, 0, sizeof file_path);
    if (!path) return 1;
    snprintf(file_path, sizeof file_path, "%s", path);
    FILE *f = fopen(path, "rb");
    if (!f) return 301;
    char buf[512];
    size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    const char *line = type == 1 ? memmem(buf, n, ".target", 7) : 0;
    for (size_t i = 0; line && i < sizeof seen_target - 1 && (size_t)(line - buf) + i < n && line[i] != '\n'; ++i)
        seen_target[i] = line[i];
    return 0;
}
int cuLinkAddFile(void *state, int type, const char *path, unsigned count, int *options, void **values) {
    return cuLinkAddFile_v2(state, type, path, count, options, values);
}
int cuLinkComplete(void *state, void **image, size_t *size) {
    const char *slow = getenv("MOCK_LOAD_SLEEP");
    if (slow) {
        fprintf(stderr, "mock: cuLinkComplete begins\n");
        usleep((useconds_t)(atof(slow) * 1e6));
        fprintf(stderr, "mock: cuLinkComplete returns\n");
    }
    *image = (void *)linked;
    if (size) *size = linked_size;
    return 0;
}
int cuLinkDestroy(void *state) {
    if (destroy_fails) return 400;
    ++destroyed;
    if (while_freed) while_freed();
    return 0;
}

static uintptr_t modules;
int cuModuleLoadData(void **module, const void *image) {
    const char *slow = getenv("MOCK_LOAD_SLEEP");
    if (slow) {
        fprintf(stderr, "mock: cuModuleLoadData begins\n");
        usleep((useconds_t)(atof(slow) * 1e6));
        fprintf(stderr, "mock: cuModuleLoadData returns\n");
    }
    *module = (void *)(next_module ? next_module : 0x100000 + 0x100 * ++modules);
    next_module = 0;
    return 0;
}
int cuModuleUnload(void *module) {
    if (unload_fails) return 201;
    if (while_freed) while_freed();
    return 0;
}
int cuModuleGetFunction(void **function, void *module, const char *name) {
    *function = (void *)((uintptr_t)module + 1);
    return 0;
}
int cuModuleLoadDataEx(void **module, const void *image, unsigned count, int *options, void **values) {
    return cuModuleLoadData(module, image);
}
int cuModuleLoadFatBinary(void **module, const void *image) { return cuModuleLoadData(module, image); }
int cuModuleLoad(void **module, const char *path) { return cuModuleLoadData(module, path); }
/* Libraries, with every kernel they are asked for. */
int cuLibraryLoadData(void **library, const void *image, int *jo, void **jv, unsigned jc, int *lo, void **lv,
                      unsigned lc) {
    return cuModuleLoadData(library, image);
}
int cuLibraryLoadFromFile(void **library, const char *path, int *jo, void **jv, unsigned jc, int *lo, void **lv,
                          unsigned lc) {
    return cuModuleLoadData(library, path);
}
int cuLibraryUnload(void *library) { return cuModuleUnload(library); }
int cuLibraryGetKernel(void **kernel, void *library, const char *name) {
    *kernel = (void *)((uintptr_t)library + 2);
    return 0;
}
