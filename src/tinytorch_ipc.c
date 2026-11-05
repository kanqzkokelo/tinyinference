#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
/*
 * tinytorch_ipc.c - Lock-free shared-memory ring buffer IPC
 *
 * Implements high-throughput (> 50,000 req/s), microsecond-latency
 * inter-process communication using memory-mapped circular queues with
 * C11 atomic head/tail pointers and memory fences.
 *
 * POSIX uses shm_open + mmap; Windows uses a named page-file mapping
 * (CreateFileMapping/MapViewOfFile) with the same layout and atomics.
 */

#include "tinytorch_ipc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <stdint.h>
#include <errno.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

typedef struct {
    _Atomic uint32_t head;
    _Atomic uint32_t tail;
    TTIpcRequest entries[TT_IPC_QUEUE_CAP];
} ReqQueue;

typedef struct {
    _Atomic uint32_t head;
    _Atomic uint32_t tail;
    TTIpcResponse entries[TT_IPC_QUEUE_CAP];
} RespQueue;

typedef struct {
    ReqQueue req_q;
    RespQueue resp_q;
    _Atomic uint32_t active_clients;
} ShmBuffer;

/* ------------------------------------------------------------------ */
/* shared-memory primitives                                            */
/* ------------------------------------------------------------------ */

#ifdef _WIN32

typedef intptr_t tt_shm_handle;
#define TT_SHM_INVALID ((tt_shm_handle)-1)
#define TT_SHM_PREFIX "tt_ipc_"

static tt_shm_handle tt_shm_create(const char *name, size_t size, void **map) {
    HANDLE h = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                                  (DWORD)((uint64_t)size >> 32),
                                  (DWORD)(size & 0xffffffffu), name);
    if (!h || GetLastError() == ERROR_ALREADY_EXISTS) {
        if (h) CloseHandle(h);
        return TT_SHM_INVALID;
    }
    void *p = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!p) {
        CloseHandle(h);
        return TT_SHM_INVALID;
    }
    *map = p;
    return (tt_shm_handle)h;
}

static tt_shm_handle tt_shm_connect(const char *name, size_t size, void **map) {
    HANDLE h = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name);
    if (!h) return TT_SHM_INVALID;
    void *p = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!p) {
        CloseHandle(h);
        return TT_SHM_INVALID;
    }
    *map = p;
    return (tt_shm_handle)h;
}

static void tt_shm_close(tt_shm_handle h, void *map, size_t size) {
    (void)size;
    if (map) UnmapViewOfFile(map);
    if (h != TT_SHM_INVALID) CloseHandle((HANDLE)h);
}

/* The kernel object is destroyed when its last handle closes. */
static void tt_shm_unlink_name(const char *name) { (void)name; }

static void tt_shm_perror(const char *what) {
    fprintf(stderr, "%s failed (win32 error %lu)\n", what, (unsigned long)GetLastError());
}

#else /* POSIX */

typedef int tt_shm_handle;
#define TT_SHM_INVALID (-1)
#define TT_SHM_PREFIX "/tt_ipc_"

static tt_shm_handle tt_shm_create(const char *name, size_t size, void **map) {
    shm_unlink(name); /* clean up a stale segment */

    int fd = shm_open(name, O_CREAT | O_RDWR | O_EXCL, 0666);
    if (fd < 0) return TT_SHM_INVALID;

    if (ftruncate(fd, (off_t)size) < 0) {
        close(fd);
        shm_unlink(name);
        return TT_SHM_INVALID;
    }

    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        close(fd);
        shm_unlink(name);
        return TT_SHM_INVALID;
    }
    *map = p;
    return fd;
}

static tt_shm_handle tt_shm_connect(const char *name, size_t size, void **map) {
    int fd = shm_open(name, O_RDWR, 0666);
    if (fd < 0) return TT_SHM_INVALID;

    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        close(fd);
        return TT_SHM_INVALID;
    }
    *map = p;
    return fd;
}

static void tt_shm_close(tt_shm_handle h, void *map, size_t size) {
    if (map) munmap(map, size);
    if (h != TT_SHM_INVALID) close(h);
}

static void tt_shm_unlink_name(const char *name) { shm_unlink(name); }

static void tt_shm_perror(const char *what) { perror(what); }

#endif /* _WIN32 */

/* ------------------------------------------------------------------ */
/* endpoints                                                           */
/* ------------------------------------------------------------------ */

struct TTIpcServer {
    char shm_name[TT_IPC_NAME_MAX];
    tt_shm_handle shm_fd;
    ShmBuffer *shm;
};

struct TTIpcClient {
    char shm_name[TT_IPC_NAME_MAX];
    tt_shm_handle shm_fd;
    ShmBuffer *shm;
};

TTIpcServer *tt_ipc_server_create(const char *name) {
    if (!name || !name[0]) return NULL;
    TTIpcServer *s = (TTIpcServer *)calloc(1, sizeof(TTIpcServer));
    if (!s) return NULL;

    snprintf(s->shm_name, sizeof(s->shm_name), TT_SHM_PREFIX "%s", name);
    s->shm_fd = tt_shm_create(s->shm_name, sizeof(ShmBuffer), (void **)&s->shm);
    if (s->shm_fd == TT_SHM_INVALID) {
        tt_shm_perror("shm create");
        free(s);
        return NULL;
    }

    memset(s->shm, 0, sizeof(ShmBuffer));
    atomic_init(&s->shm->req_q.head, 0);
    atomic_init(&s->shm->req_q.tail, 0);
    atomic_init(&s->shm->resp_q.head, 0);
    atomic_init(&s->shm->resp_q.tail, 0);
    atomic_init(&s->shm->active_clients, 0);

    return s;
}

bool tt_ipc_server_recv_request(TTIpcServer *server, TTIpcRequest *req) {
    if (!server || !server->shm || !req) return false;
    ReqQueue *q = &server->shm->req_q;

    uint32_t head = atomic_load_explicit(&q->head, memory_order_relaxed);
    uint32_t tail = atomic_load_explicit(&q->tail, memory_order_acquire);

    if (head == tail) {
        return false; // Empty queue
    }

    *req = q->entries[head % TT_IPC_QUEUE_CAP];
    atomic_store_explicit(&q->head, head + 1, memory_order_release);
    return true;
}

bool tt_ipc_server_send_response(TTIpcServer *server, const TTIpcResponse *resp) {
    if (!server || !server->shm || !resp) return false;
    RespQueue *q = &server->shm->resp_q;

    uint32_t head = atomic_load_explicit(&q->head, memory_order_acquire);
    uint32_t tail = atomic_load_explicit(&q->tail, memory_order_relaxed);

    if (tail - head >= TT_IPC_QUEUE_CAP) {
        return false; // Queue full
    }

    q->entries[tail % TT_IPC_QUEUE_CAP] = *resp;
    atomic_store_explicit(&q->tail, tail + 1, memory_order_release);
    return true;
}

void tt_ipc_server_destroy(TTIpcServer *server) {
    if (!server) return;
    tt_shm_close(server->shm_fd, server->shm, sizeof(ShmBuffer));
    tt_shm_unlink_name(server->shm_name);
    free(server);
}

TTIpcClient *tt_ipc_client_connect(const char *name) {
    if (!name || !name[0]) return NULL;
    TTIpcClient *c = (TTIpcClient *)calloc(1, sizeof(TTIpcClient));
    if (!c) return NULL;

    snprintf(c->shm_name, sizeof(c->shm_name), TT_SHM_PREFIX "%s", name);
    c->shm_fd = tt_shm_connect(c->shm_name, sizeof(ShmBuffer), (void **)&c->shm);
    if (c->shm_fd == TT_SHM_INVALID) {
        free(c);
        return NULL;
    }

    atomic_fetch_add_explicit(&c->shm->active_clients, 1, memory_order_relaxed);
    return c;
}

bool tt_ipc_client_send_request(TTIpcClient *client, const TTIpcRequest *req) {
    if (!client || !client->shm || !req) return false;
    ReqQueue *q = &client->shm->req_q;

    uint32_t head = atomic_load_explicit(&q->head, memory_order_acquire);
    uint32_t tail = atomic_load_explicit(&q->tail, memory_order_relaxed);

    if (tail - head >= TT_IPC_QUEUE_CAP) {
        return false; // Queue full
    }

    q->entries[tail % TT_IPC_QUEUE_CAP] = *req;
    atomic_store_explicit(&q->tail, tail + 1, memory_order_release);
    return true;
}

bool tt_ipc_client_recv_response(TTIpcClient *client, TTIpcResponse *resp) {
    if (!client || !client->shm || !resp) return false;
    RespQueue *q = &client->shm->resp_q;

    uint32_t head = atomic_load_explicit(&q->head, memory_order_relaxed);
    uint32_t tail = atomic_load_explicit(&q->tail, memory_order_acquire);

    if (head == tail) {
        return false; // Empty queue
    }

    *resp = q->entries[head % TT_IPC_QUEUE_CAP];
    atomic_store_explicit(&q->head, head + 1, memory_order_release);
    return true;
}

void tt_ipc_client_disconnect(TTIpcClient *client) {
    if (!client) return;
    if (client->shm) {
        atomic_fetch_sub_explicit(&client->shm->active_clients, 1, memory_order_relaxed);
    }
    tt_shm_close(client->shm_fd, client->shm, sizeof(ShmBuffer));
    free(client);
}
