#include <stdio.h>
#include <string.h>
#include "loader_gguf.h"
int main(int argc, char **argv) {
  GGUFModel *m = gguf_load(argv[1]);
  if (!m) { printf("load fail\n"); return 1; }
  for (int i = 0; i < m->tensor_count; i++) {
    const char *n = m->tensors[i].name;
    if (strstr(n, "weight") || strstr(n, "output"))
      printf("%s type=%d ndim=%d\n", n, (int)m->tensors[i].type, m->tensors[i].ndim);
  }
  return 0;
}
