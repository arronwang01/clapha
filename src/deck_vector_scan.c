// Deck-vector scan, in the same idiom as the repo's mumu_cycle_vector_scan.c:
// walk only the game's own scudo: mappings, look for a std::vector-shaped header
// {data pointer, capacity, size}, validate what it points at, and report.
//
// mumu_cycle_vector_scan looks for size==4, cap 4..8, values 0..7 distinct  -> hand/cycle.
// A deck is the same shape with size==8. Two payload forms are accepted:
//   A) eight int32 card ids
//   B) eight pointers, each to an object carrying a card id at +0x40 (the owner/entry
//      layout the repo's read_visible_deck walks: entry+0x10 -> data -> +0x40)
//
// Read-only. No whole-heap sweep, no dalvik spaces, no pointer chasing beyond the payload.
//
// usage: deck_vector_scan PID
#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_MAPS 16384
#define MAX_HITS 2048
#define UNTAG(p) ((p) & 0x00FFFFFFFFFFFFFFULL)
#define DECK 8

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

/* table*1000000 + row, small row: base 26/27/28, evolution 13, hero 203 */
static int card(int32_t v) {
  int table = v / 1000000, row = v % 1000000;
  if (row < 0 || row > 300) return 0;
  return table == 26 || table == 27 || table == 28 || table == 13 || table == 203;
}

static int eight(const int32_t *v) {
  for (int i = 0; i < DECK; ++i) {
    if (!card(v[i])) return 0;
    for (int k = 0; k < i; ++k) if (v[k] == v[i]) return 0;
  }
  /* a run like 26000000,26000001,... is the card table, not a deck */
  int run = 1;
  for (int i = 1; i < DECK; ++i) if (v[i] != v[i - 1] + 1) run = 0;
  return !run;
}

int main(int ac, char **av) {
  if (ac != 2) { fprintf(stderr, "usage: deck_vector_scan PID\n"); return 2; }
  int pid = atoi(av[1]);
  Map mm[MAX_MAPS];
  int mc = maps(pid, mm);
  if (mc <= 0) return 3;
  char p[64];
  snprintf(p, sizeof(p), "/proc/%d/mem", pid);
  int f = open(p, O_RDONLY | O_CLOEXEC);
  if (f < 0) return 4;

  printf("{\"event\":\"deck_vector_scan\",\"pid\":%d,\"hits\":[", pid);
  int out = 0, headers = 0;
  uint64_t scanned = 0;
  for (int mi = 0; mi < mc && out < MAX_HITS; ++mi) {
    Map *m = &mm[mi];
    if (!m->r || !m->w || !strstr(m->p, "scudo:")) continue;
    for (uint64_t s = m->s; s < m->e && out < MAX_HITS; s += 0x1000) {
      size_t n = (size_t)((m->e - s) < 0x1000 ? (m->e - s) : 0x1000);
      uint8_t *b = malloc(n);
      if (!b || !rd(f, s, b, n)) { free(b); continue; }
      scanned += n;
      for (size_t o = 0; o + 16 <= n && out < MAX_HITS; o += 8) {
        uint64_t data = 0;
        int32_t at8 = 0, at12 = 0;
        memcpy(&data, b + o, 8);
        memcpy(&at8, b + o + 8, 4);
        memcpy(&at12, b + o + 12, 4);
        /* Two header shapes occur here:
             {data, capacity, size}   - std::vector, as mumu_cycle_vector_scan assumes
             {data, ..., count}       - the owner deck: entries at +0x20, count at +0x2c
           so accept a count of 8 in either word and do not require a capacity. */
        int32_t cap = at8;
        if (!(at12 == DECK || at8 == DECK) || !data) continue;
        if (UNTAG(data) < 0x100000) continue;
        ++headers;

        int32_t flat[DECK];
        int form = 0;
        if (rd(f, data, flat, sizeof(flat)) && eight(flat)) {
          form = 1;                                   /* A: eight card ids */
        } else {
          uint64_t pointers[DECK];
          if (!rd(f, data, pointers, sizeof(pointers))) continue;
          int ok = 1;
          for (int k = 0; k < DECK && ok; ++k) {
            uint64_t entry = pointers[k], inner = 0;
            int32_t value = -1;
            if (!entry || UNTAG(entry) < 0x100000) { ok = 0; break; }
            /* entry+0x10 -> data -> +0x40, as read_visible_deck walks it */
            if (rd(f, entry + 0x10, &inner, 8) && inner &&
                rd(f, inner + 0x40, &value, 4) && card(value)) {
              flat[k] = value;
            } else if (rd(f, entry + 0x40, &value, 4) && card(value)) {
              flat[k] = value;
            } else {
              ok = 0;
            }
          }
          if (ok && eight(flat)) form = 2;             /* B: eight entry pointers */
        }
        if (!form) continue;

        if (out++) putchar(',');
        printf("{\"header\":\"0x%" PRIx64 "\",\"data\":\"0x%" PRIx64 "\",\"cap\":%d,"
               "\"form\":\"%s\",\"map\":\"%s\",\"cards\":[",
               s + o, data, cap, form == 1 ? "card_ids" : "entry_pointers", m->p);
        for (int k = 0; k < DECK; ++k) printf("%s%d", k ? "," : "", flat[k]);
        printf("]}");
      }
      free(b);
    }
  }
  printf("],\"hit_count\":%d,\"headers_examined\":%d,\"bytes_scanned\":%" PRIu64 "}\n",
         out, headers, scanned);
  close(f);
  return 0;
}
