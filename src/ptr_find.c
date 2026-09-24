// Find who points at a given address, in the repo's scanning idiom:
// walk only the game's own scudo: mappings and report every 8-byte slot whose value,
// with the Android heap tag masked off, equals the target. Read-only.
//
// Used to turn a scan hit into a pointer chain: given the address of a structure found by
// deck_vector_scan, find its parent, then repeat.
//
// With --range LO HI it reports any pointer landing inside [LO,HI], which finds a parent
// that points at some internal offset of the object rather than its base. With --all it
// scans every writable mapping, not just scudo: (the referrer may live in libg's .bss or
// the dalvik heap).
//
// usage: ptr_find PID TARGET [MORE...]
//        ptr_find PID --range LO HI [--all]
#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_MAPS 16384
#define MAX_TARGETS 8
#define MAX_HITS 512
#define UNTAG(p) ((p) & 0x00FFFFFFFFFFFFFFULL)

typedef struct { uint64_t s, e; int r, w; char p[128]; } Map;

static int rd(int f, uint64_t a, void *o, size_t s) {
  uint8_t *p = o;
  size_t n = 0;
  while (n < s) {
    ssize_t v = pread(f, p + n, s - n, (off_t)(UNTAG(a) + n));
    if (v <= 0) return 0;
    n += (size_t)v;
  }
  return 1;
}

static int maps(int pid, Map *m) {
  char f[64], l[1024];
  snprintf(f, sizeof(f), "/proc/%d/maps", pid);
  FILE *h = fopen(f, "r");
  if (!h) return -1;
  int c = 0;
  while (c < MAX_MAPS && fgets(l, sizeof(l), h)) {
    unsigned long long s, e, o;
    unsigned int a, b;
    unsigned long ino;
    char q[8];
    int u = 0;
    if (sscanf(l, "%llx-%llx %7s %llx %x:%x %lu %n", &s, &e, q, &o, &a, &b, &ino, &u) < 7)
      continue;
    Map *x = &m[c++];
    memset(x, 0, sizeof(*x));
    x->s = s; x->e = e; x->r = q[0] == 'r'; x->w = q[1] == 'w';
    char *n = l + u;
    while (*n == ' ' || *n == '\t') ++n;
    size_t z = strcspn(n, "\r\n");
    if (z >= sizeof(x->p)) z = sizeof(x->p) - 1;
    memcpy(x->p, n, z);
  }
  fclose(h);
  return c;
}

int main(int ac, char **av) {
  if (ac < 3 || ac > 2 + MAX_TARGETS) {
    fprintf(stderr, "usage: ptr_find PID TARGET [MORE...]\n");
    return 2;
  }
  int pid = atoi(av[1]);
  uint64_t targets[MAX_TARGETS];
  int target_count = 0, all_maps = 0;
  uint64_t lo = 0, hi = 0;
  for (int i = 2; i < ac; ++i) {
    if (!strcmp(av[i], "--all")) { all_maps = 1; continue; }
    if (!strcmp(av[i], "--range") && i + 2 < ac) {
      lo = UNTAG(strtoull(av[i + 1], NULL, 0));
      hi = UNTAG(strtoull(av[i + 2], NULL, 0));
      i += 2;
      continue;
    }
    if (target_count < MAX_TARGETS)
      targets[target_count++] = UNTAG(strtoull(av[i], NULL, 0));
  }
  if (!target_count && !hi) { fprintf(stderr, "no targets\n"); return 2; }

  Map *mm = calloc(MAX_MAPS, sizeof(Map));
  int mc = maps(pid, mm);
  if (mc <= 0) return 3;
  char p[64];
  snprintf(p, sizeof(p), "/proc/%d/mem", pid);
  int f = open(p, O_RDONLY | O_CLOEXEC);
  if (f < 0) return 4;

  printf("{\"event\":\"ptr_find\",\"pid\":%d,\"hits\":[", pid);
  int out = 0;
  uint64_t scanned = 0;
  uint8_t *b = malloc(0x10000);
  for (int mi = 0; mi < mc && out < MAX_HITS; ++mi) {
    Map *m = &mm[mi];
    if (!m->r || !m->w) continue;
    if (!all_maps && !strstr(m->p, "scudo:")) continue;
    if (all_maps && (strstr(m->p, "/dev/") || strstr(m->p, "[vvar"))) continue;
    for (uint64_t s = m->s; s < m->e && out < MAX_HITS; s += 0x10000) {
      size_t n = (size_t)((m->e - s) < 0x10000 ? (m->e - s) : 0x10000);
      if (!rd(f, s, b, n)) continue;
      scanned += n;
      for (size_t o = 0; o + 8 <= n && out < MAX_HITS; o += 8) {
        uint64_t value;
        memcpy(&value, b + o, 8);
        if (!value) continue;
        uint64_t clean = UNTAG(value);
        if (hi && clean >= lo && clean <= hi) {
          if (s + o >= lo && s + o <= hi) continue;      /* ignore self-references */
          if (out++) putchar(',');
          printf("{\"at\":\"0x%" PRIx64 "\",\"raw\":\"0x%" PRIx64 "\",\"points_to\":\"0x%"
                 PRIx64 "\",\"map\":\"%s\"}", s + o, value, clean, m->p);
          continue;
        }
        for (int t = 0; t < target_count; ++t) {
          if (clean != targets[t]) continue;
          if (out++) putchar(',');
          printf("{\"at\":\"0x%" PRIx64 "\",\"raw\":\"0x%" PRIx64 "\",\"target\":\"0x%" PRIx64
                 "\",\"map\":\"%s\"}", s + o, value, targets[t], m->p);
          break;
        }
      }
    }
  }
  printf("],\"hit_count\":%d,\"bytes_scanned\":%" PRIu64 "}\n", out, scanned);
  free(b);
  free(mm);
  close(f);
  return 0;
}
