// Report every address holding a given int32 value. Used for value-pair differentials:
// a field encoding the hand as ONE word (bitmask, packed nibbles, permutation index)
// cannot be found by any search for four adjacent slots, but it must change from
// encode(H1) to encode(H2) at the same address. Intersecting the two passes finds it.
// usage: value_scan PID VALUE [--all]
#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(int ac, char **av) {
  if (ac < 3) return 2;
  int pid = atoi(av[1]);
  int32_t want = (int32_t)strtol(av[2], NULL, 0);
  int all_maps = (ac > 3 && !strcmp(av[3], "--all"));
  char f[64], l[1024];
  snprintf(f, sizeof(f), "/proc/%d/maps", pid);
  FILE *h = fopen(f, "r"); if (!h) return 3;
  snprintf(f, sizeof(f), "/proc/%d/mem", pid);
  int fd = open(f, O_RDONLY | O_CLOEXEC); if (fd < 0) return 4;
  printf("{\"value\":%d,\"at\":[", want);
  int out = 0; uint64_t scanned = 0;
  uint8_t *b = malloc(0x10000);
  while (fgets(l, sizeof(l), h)) {
    unsigned long long s, e; char q[8];
    if (sscanf(l, "%llx-%llx %7s", &s, &e, q) != 3) continue;
    if (q[0] != 'r' || q[1] != 'w') continue;
    if (!all_maps && !strstr(l, "scudo:")) continue;
    if (strstr(l, "/dev/")) continue;
    for (uint64_t at = s; at < e; at += 0x10000) {
      size_t n = (size_t)((e - at) < 0x10000 ? (e - at) : 0x10000);
      if (pread(fd, b, n, (off_t)at) != (ssize_t)n) continue;
      scanned += n;
      int32_t *w = (int32_t *)b;
      for (size_t i = 0; i < n / 4; ++i)
        if (w[i] == want) { if (out++) putchar(','); printf("\"0x%" PRIx64 "\"", at + i * 4); }
    }
  }
  printf("],\"count\":%d,\"bytes\":%" PRIu64 "}\n", out, scanned);
  return 0;
}
