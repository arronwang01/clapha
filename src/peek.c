// Minimal read-only peek: dump N bytes at an address as int32 and int64 columns.
// usage: peek PID ADDR [BYTES]
#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define UNTAG(p) ((p) & 0x00FFFFFFFFFFFFFFULL)
int main(int ac, char **av) {
  if (ac < 3) { fprintf(stderr, "usage: peek PID ADDR [BYTES]\n"); return 2; }
  int pid = atoi(av[1]);
  uint64_t addr = strtoull(av[2], NULL, 0);
  size_t size = ac > 3 ? (size_t)strtoul(av[3], NULL, 0) : 0x120;
  char path[64];
  snprintf(path, sizeof(path), "/proc/%d/mem", pid);
  int f = open(path, O_RDONLY | O_CLOEXEC);
  if (f < 0) return 3;
  uint8_t *b = malloc(size);
  if (pread(f, b, size, (off_t)UNTAG(addr)) != (ssize_t)size) { perror("pread"); return 4; }
  printf("{\"addr\":\"0x%" PRIx64 "\",\"words\":[", UNTAG(addr));
  for (size_t o = 0; o + 4 <= size; o += 4) {
    int32_t v; memcpy(&v, b + o, 4);
    printf("%s[\"0x%zx\",%d]", o ? "," : "", o, v);
  }
  printf("],\"pointers\":[");
  int first = 1;
  for (size_t o = 0; o + 8 <= size; o += 8) {
    uint64_t v; memcpy(&v, b + o, 8);
    if (UNTAG(v) < 0x100000 || UNTAG(v) > 0x0000800000000000ULL) continue;
    printf("%s[\"0x%zx\",\"0x%" PRIx64 "\"]", first ? "" : ",", o, v);
    first = 0;
  }
  printf("]}\n");
  return 0;
}
