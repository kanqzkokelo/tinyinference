#include "async_printer.h"
#include "tt_platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#define ASYNC_BUF_SIZE 16384
#define ASYNC_QUEUE_CAP 1024

struct AsyncPrinter {
    const char *queue_str[ASYNC_QUEUE_CAP];
    int queue_len[ASYNC_QUEUE_CAP];
    _Atomic int head;
    _Atomic int tail;
    _Atomic int done;
#ifdef _WIN32
    HANDLE thread;
#else
    pthread_t thread;
#endif
};

static void ap_write(const char *buf, int len) {
#ifdef _WIN32
    fwrite(buf, 1, (size_t)len, stdout);
    fflush(stdout);
#else
    if (write(STDOUT_FILENO, buf, (size_t)len) < 0) {}
#endif
}

#ifdef _WIN32
static DWORD WINAPI async_printer_worker(LPVOID arg) {
#else
static void *async_printer_worker(void *arg) {
#endif
    AsyncPrinter *ap = (AsyncPrinter *)arg;
    char io_buf[ASYNC_BUF_SIZE];
    int buf_len = 0;

    while (!atomic_load(&ap->done) || atomic_load(&ap->head) != atomic_load(&ap->tail)) {
        int head = atomic_load(&ap->head);
        int tail = atomic_load(&ap->tail);

        if (head != tail) {
            const char *str = ap->queue_str[head];
            int len = ap->queue_len[head];
            atomic_store(&ap->head, (head + 1) % ASYNC_QUEUE_CAP);

            if (str && len > 0) {
                if (buf_len + len < (int)sizeof(io_buf)) {
                    memcpy(io_buf + buf_len, str, (size_t)len);
                    buf_len += len;
                } else {
                    ap_write(io_buf, buf_len);
                    buf_len = 0;
                    memcpy(io_buf + buf_len, str, (size_t)len);
                    buf_len += len;
                }
            }
        } else {
            if (buf_len > 0) {
                ap_write(io_buf, buf_len);
                buf_len = 0;
            }
            tt_sleep_us(100);
        }
    }

    if (buf_len > 0) ap_write(io_buf, buf_len);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

AsyncPrinter *async_printer_start(void) {
    AsyncPrinter *ap = (AsyncPrinter *)calloc(1, sizeof(AsyncPrinter));
    if (!ap) return NULL;
    atomic_store(&ap->head, 0);
    atomic_store(&ap->tail, 0);
    atomic_store(&ap->done, 0);

#ifdef _WIN32
    ap->thread = CreateThread(NULL, 0, async_printer_worker, ap, 0, NULL);
    if (!ap->thread) { free(ap); return NULL; }
#else
    pthread_create(&ap->thread, NULL, async_printer_worker, ap);
#endif
    return ap;
}

void async_printer_push(AsyncPrinter *ap, const char *str, int len) {
    if (!ap || !str || len <= 0) return;
    int tail = atomic_load(&ap->tail);
    ap->queue_str[tail] = str;
    ap->queue_len[tail] = len;
    atomic_store(&ap->tail, (tail + 1) % ASYNC_QUEUE_CAP);
}

void async_printer_stop_and_flush(AsyncPrinter *ap) {
    if (!ap) return;
    atomic_store(&ap->done, 1);
#ifdef _WIN32
    WaitForSingleObject(ap->thread, INFINITE);
    CloseHandle(ap->thread);
#else
    pthread_join(ap->thread, NULL);
#endif
    free(ap);
}
