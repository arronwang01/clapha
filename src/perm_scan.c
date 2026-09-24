// Scan scudo: mappings for 8 consecutive int32 forming a permutation of 0..7 (a full
// hand+cycle ordering stored inline, with no vector header). Same idiom as the repo's
// scanners: bounded maps, 4 KB chunks, structural validation. Read-only.
// With --all it scans every writable mapping, not just scudo: (the scudo-only filter is
// inherited from the repo, which targeted a translated Windows MuMu; on native arm64 the
// data may live elsewhere). With --seq a,b,c,d it also reports that exact int32 run.
// usage: perm_scan PID [--all] [--seq a,b,c,d]
#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define MAX_MAPS 16384
int main(int ac, char **av) {
  if (ac < 2) return 2;
  int pid = atoi(av[1]);
  int all_maps = 0; int32_t seq[4]; int have_seq = 0, as_set = 0;
  for (int i = 2; i < ac; ++i) {
    if (!strcmp(av[i], "--all")) all_maps = 1;
    else if (!strcmp(av[i], "--set")) as_set = 1;
    else if (!strcmp(av[i], "--seq") && i + 1 < ac) {
      have_seq = sscanf(av[i+1], "%d,%d,%d,%d", &seq[0], &seq[1], &seq[2], &seq[3]) == 4;
      ++i;
    }
  }
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
    if (all_maps && (strstr(l, "/dev/") || strstr(l, "[vvar"))) continue;
    char tag[96] = {0};
    { char *t = strchr(l, '['); if (!t) t = strchr(l, '/');
      if (t) { size_t z = strcspn(t, "\r\n"); if (z > 95) z = 95; memcpy(tag, t, z); } }
    for (uint64_t at = s; at < e; at += 0x1000) {
      size_t n = (size_t)((e - at) < 0x1000 ? (e - at) : 0x1000);
      if (pread(fd, b, n, (off_t)at) != (ssize_t)n) continue;
      scanned += n;
      int32_t *w = (int32_t *)b;
      if (have_seq) {
        for (size_t i = 0; i + 4 <= n / 4; ++i) {
          int match;
          if (as_set) {   /* the four values in any order, e.g. a hand however it is stored */
            int mask = 0, want = 0, ok = 1;
            for (int k = 0; k < 4; ++k) want |= 1 << seq[k];
            for (int k = 0; k < 4 && ok; ++k) {
              int32_t v = w[i+k];
              if (v < 0 || v > 7 || (mask & (1 << v))) ok = 0; else mask |= 1 << v;
            }
            match = ok && mask == want;
          } else {
            match = (w[i] == seq[0] && w[i+1] == seq[1] && w[i+2] == seq[2] && w[i+3] == seq[3]);
          }
          if (match) {
            if (out++) putchar(',');
            printf("{\"at\":\"0x%" PRIx64 "\",\"kind\":\"seq\",\"map\":\"%s\",\"order\":[%d,%d,%d,%d]}",
                   at + i * 4, tag, w[i], w[i+1], w[i+2], w[i+3]);
          }
        }
      }
      for (size_t i = 0; i + 8 <= n / 4; ++i) {
        int mask = 0, ok = 1;
        for (int k = 0; k < 8 && ok; ++k) {
          int32_t v = w[i + k];
          if (v < 0 || v > 7 || (mask & (1 << v))) ok = 0; else mask |= 1 << v;
        }
        if (!ok || mask != 0xff) continue;
        if (out++) putchar(',');
        printf("{\"at\":\"0x%" PRIx64 "\",\"kind\":\"perm\",\"map\":\"%s\",\"order\":[", at + i * 4, tag);
        for (int k = 0; k < 8; ++k) printf("%s%d", k ? "," : "", w[i + k]);
        printf("]}");
      }
    }
  }
  printf("],\"hit_count\":%d,\"bytes_scanned\":%" PRIu64 "}\n", out, scanned);
  return 0;
}
