/*
 * tt_platform.h - tiny portability shims for the portable C surface.
 *
 * Linux/macOS keep the original POSIX calls.  Windows uses Win32 directly
 * (no mingw/Cygwin runtime dependency for the mapped-file and env paths).
 *
 * Header-only on purpose: this file is included by sources that are built
 * both into the shared library and standalone test binaries.
 */
#ifndef TT_PLATFORM_H
#define TT_PLATFORM_H

#include <stddef.h>

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* POSIX clock_gettime(CLOCK_MONOTONIC) used by the timing code in examples
 * and tools.  MSVC has struct timespec but no clock_gettime; fill it from
 * QueryPerformanceCounter.  Deltas are all that callers consume. */
#ifdef _MSC_VER
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
static inline int tt_clock_gettime(int clk, struct timespec *ts) {
    (void)clk;
    static LARGE_INTEGER freq;
    LARGE_INTEGER c;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    LONGLONG whole = c.QuadPart / freq.QuadPart;
    LONGLONG rem = c.QuadPart % freq.QuadPart;
    ts->tv_sec = (time_t)whole;
    ts->tv_nsec = (long)((rem * 1000000000LL) / freq.QuadPart);
    return 0;
}
#define clock_gettime tt_clock_gettime
#define strdup _strdup
#define strtok_r strtok_s

static inline char *tt_strcasestr(const char *haystack, const char *needle) {
    if (!*needle) return (char *)haystack;
    for (const char *h = haystack; *h; h++) {
        const char *a = h, *b = needle;
        while (*a && *b &&
               tolower((unsigned char)*a) == tolower((unsigned char)*b)) {
            a++;
            b++;
        }
        if (!*b) return (char *)h;
    }
    return NULL;
}
#define strcasestr tt_strcasestr
#endif

/* Minimal dlopen/dlsym shim: qwen2_cuda loads the optional cuBLAS helper. */
static inline void *tt_dlopen(const char *path, int flags) {
    (void)flags;
    return (void *)LoadLibraryA(path);
}
static inline void *tt_dlsym(void *handle, const char *name) {
    return (void *)GetProcAddress((HMODULE)handle, name);
}
static inline int tt_dlclose(void *handle) {
    return FreeLibrary((HMODULE)handle) ? 0 : -1;
}
static inline const char *tt_dlerror(void) {
    static char buf[64];
    snprintf(buf, sizeof(buf), "win32 error %lu", (unsigned long)GetLastError());
    return buf;
}

/* Map a whole file read-only.  Returns NULL on failure and stores the byte
 * size in *size.  The mapping stays valid after the file handle is closed;
 * only the view needs to be released with tt_unmap_file(). */
static inline void *tt_map_file(const char *path, size_t *size) {
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) return NULL;

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz) || sz.QuadPart <= 0) {
        CloseHandle(fh);
        return NULL;
    }

    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(fh); /* the mapping object keeps the file open */
    if (!mh) return NULL;

    void *view = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mh); /* the view keeps its own reference */
    if (!view) return NULL;

    *size = (size_t)sz.QuadPart;
    return view;
}

static inline void tt_unmap_file(void *addr, size_t size) {
    (void)size;
    if (addr) UnmapViewOfFile(addr);
}

/* POSIX setenv/unsetenv equivalents.  _putenv_s("NAME","") removes NAME. */
static inline int tt_setenv(const char *name, const char *value, int overwrite) {
    if (!overwrite) {
        DWORD n = GetEnvironmentVariableA(name, NULL, 0);
        if (n != 0) return 0;
    }
    return _putenv_s(name, value ? value : "");
}

static inline int tt_unsetenv(const char *name) {
    return _putenv_s(name, "");
}
#define setenv tt_setenv
#define unsetenv tt_unsetenv

/* Monotonic milliseconds since an arbitrary epoch. */
static inline double tt_now_ms(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER c;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
}

static inline double tt_now_us(void) { return tt_now_ms() * 1000.0; }

static inline void tt_sleep_ms(int ms) { Sleep(ms > 0 ? (DWORD)ms : 0); }
static inline void tt_sleep_us(int us) { Sleep(us > 0 ? (DWORD)((us + 999) / 1000) : 0); }

#else /* POSIX */

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static inline void *tt_map_file(const char *path, size_t *size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0 || st.st_size <= 0) {
        close(fd);
        return NULL;
    }

    size_t n = (size_t)st.st_size;
    void *p = mmap(NULL, n, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return NULL;

    *size = n;
    return p;
}

static inline void tt_unmap_file(void *addr, size_t size) {
    if (addr) munmap(addr, size);
}

static inline int tt_setenv(const char *name, const char *value, int overwrite) {
    return setenv(name, value, overwrite);
}

static inline int tt_unsetenv(const char *name) { return unsetenv(name); }

static inline double tt_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static inline double tt_now_us(void) { return tt_now_ms() * 1000.0; }

static inline void tt_sleep_ms(int ms) {
    struct timespec req;
    req.tv_sec = ms / 1000;
    req.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&req, NULL);
}

static inline void tt_sleep_us(int us) {
    struct timespec req;
    req.tv_sec = us / 1000000;
    req.tv_nsec = (long)(us % 1000000) * 1000L;
    nanosleep(&req, NULL);
}

#endif /* _WIN32 */

#endif /* TT_PLATFORM_H */
