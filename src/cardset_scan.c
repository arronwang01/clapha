// Find four adjacent int32 whose values are exactly a given set of card ids.
// Same idiom as the repo's scanners: bounded maps, 4 KB chunks, structural match.
// Card ids are distinctive, so this has far less noise than scanning for slot indices.
// usage: cardset_scan PID a,b,c,d [--all]
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
  int32_t want[4];
  if (sscanf(av[2], "%d,%d,%d,%d", &want[0], &want[1], &want[2], &want[3]) != 4) return 2;
  int all_maps = (ac > 3 && !strcmp(av[3], "--all"));
  char f[64], l[1024];
  snprintf(f, sizeof(f), "/proc/%d/maps", pid);
  FILE *h = fopen(f, "r"); if (!h) return 3;
  snprintf(f, sizeof(f), "/proc/%d/mem", pid);
  int fd = open(f, O_RDONLY | O_CLOEXEC); if (fd < 0) return 4;
  printf("{\"hits\":[");
  int out = 0; uint64_t scanned = 0;
  uint8_t *b = malloc(0x1000);
  while (fgets(l, sizeof(l), h)) {
    unsigned long long s, e; char q[8];
    if (sscanf(l, "%llx-%llx %7s", &s, &e, q) != 3) continue;
    if (q[0] != 'r' || q[1] != 'w') continue;
    if (!all_maps && !strstr(l, "scudo:")) continue;
    if (strstr(l, "/dev/")) continue;
    char tag[96] = {0};
    { char *t = strchr(l, '['); if (!t) t = strchr(l, '/');
      if (t) { size_t z = strcspn(t, "\r\n"); if (z > 95) z = 95; memcpy(tag, t, z); } }
    for (uint64_t at = s; at < e; at += 0x1000) {
      size_t n = (size_t)((e - at) < 0x1000 ? (e - at) : 0x1000);
      if (pread(fd, b, n, (off_t)at) != (ssize_t)n) continue;
      scanned += n;
      int32_t *w = (int32_t *)b;
      for (size_t i = 0; i + 4 <= n / 4; ++i) {
        int used = 0, ok = 1;
        for (int k = 0; k < 4 && ok; ++k) {
          int found = -1;
          for (int j = 0; j < 4; ++j)
            if (!(used & (1 << j)) && w[i + k] == want[j]) { found = j; break; }
          if (found < 0) ok = 0; else used |= 1 << found;
        }
        if (!ok) continue;
        if (out++) putchar(',');
        printf("{\"at\":\"0x%" PRIx64 "\",\"map\":\"%s\",\"order\":[%d,%d,%d,%d]}",
               at + i * 4, tag, w[i], w[i+1], w[i+2], w[i+3]);
      }
    }
  }
  printf("],\"hit_count\":%d,\"bytes_scanned\":%" PRIu64 "}\n", out, scanned);
  return 0;
}
