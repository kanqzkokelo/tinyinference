/*
 * tt_pthread.h - the small pthread subset the two HTTP servers use.
 *
 * The portable library keeps pthreads on POSIX.  MSVC has no pthreads, so
 * Windows servers use a CRITICAL_SECTION mutex and detached Win32 threads.
 * Only create/detach/init/lock/unlock/destroy are provided.
 */
#ifndef TT_PTHREAD_H
#define TT_PTHREAD_H

#ifdef _WIN32

#include <windows.h>
#include <stdlib.h>

typedef CRITICAL_SECTION pthread_mutex_t;
typedef HANDLE pthread_t;

static inline int pthread_mutex_init(pthread_mutex_t *m, const void *attr) {
    (void)attr;
    InitializeCriticalSection(m);
    return 0;
}
static inline int pthread_mutex_lock(pthread_mutex_t *m) {
    EnterCriticalSection(m);
    return 0;
}
static inline int pthread_mutex_unlock(pthread_mutex_t *m) {
    LeaveCriticalSection(m);
    return 0;
}
static inline int pthread_mutex_destroy(pthread_mutex_t *m) {
    DeleteCriticalSection(m);
    return 0;
}

typedef struct {
    void *(*fn)(void *);
    void *arg;
} tt_thread_start;

static DWORD WINAPI tt_thread_main(LPVOID p) {
    tt_thread_start *s = (tt_thread_start *)p;
    void *(*fn)(void *) = s->fn;
    void *arg = s->arg;
    free(s);
    fn(arg);
    return 0;
}

static inline int pthread_create(pthread_t *t, const void *attr,
                                 void *(*fn)(void *), void *arg) {
    (void)attr;
    tt_thread_start *s = (tt_thread_start *)malloc(sizeof(*s));
    if (!s) return -1;
    s->fn = fn;
    s->arg = arg;
    *t = CreateThread(NULL, 0, tt_thread_main, s, 0, NULL);
    if (!*t) { free(s); return -1; }
    return 0;
}
static inline int pthread_detach(pthread_t t) {
    CloseHandle(t);
    return 0;
}

#endif /* _WIN32 */

#endif /* TT_PTHREAD_H */
