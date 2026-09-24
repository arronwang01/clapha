// Deep but bounded snapshot of one player object: the object itself, plus 0x80 bytes at
// every plausible pointer it holds (one level down). Collect once, analyse offline.
//
// Purpose: find where the opponent's live hand lives, without guessing offsets. A hand is
// 4 int32 deck slots in 0..7, so any such array reachable from the player object shows up
// in the dump, wherever it is.
//
// usage: snapshot_deep PID MANAGER_RVA CTX_OFF PLAYER_INDEX
#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define UNTAG(p) ((p) & 0x00FFFFFFFFFFFFFFULL)
#define OBJECT 0x600
#define LEAF 0x80

static int fd;

static int rd(uint64_t a, void *o, size_t n) {
  return pread(fd, o, n, (off_t)UNTAG(a)) == (ssize_t)n;
}
static uint64_t u64(uint64_t a) { uint64_t v = 0; rd(a, &v, 8); return v; }

static int plausible(uint64_t p) {
  uint64_t q = UNTAG(p);
  return q >= 0x10000 && q < 0x0000800000000000ULL && (q & 3) == 0;
}

static uint64_t libg_base(int pid) {
  char path[64], line[1024];
  snprintf(path, sizeof(path), "/proc/%d/maps", pid);
  FILE *m = fopen(path, "r");
  if (!m) return 0;
  uint64_t best = UINT64_MAX;
  while (fgets(line, sizeof(line), m)) {
    unsigned long long s = 0, off = 0; char p[8] = {0};
    if (!strstr(line, "/libg.so")) continue;
    if (sscanf(line, "%llx-%*llx %7s %llx", &s, p, &off) != 3) continue;
    if (s >= off && s - off < best) best = s - off;
  }
  fclose(m);
  return best == UINT64_MAX ? 0 : best;
}

static void hex(const uint8_t *buffer, size_t size) {
  for (size_t i = 0; i < size; ++i) printf("%02x", buffer[i]);
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: snapshot_deep PID MANAGER_RVA CTX_OFF PLAYER_INDEX\n");
    return 2;
  }
  int pid = atoi(argv[1]);
  uint64_t rva = strtoull(argv[2], NULL, 0), ctx_off = strtoull(argv[3], NULL, 0);
  int which = atoi(argv[4]);
  uint64_t base = libg_base(pid);
  if (!base) return 3;
  char path[64];
  snprintf(path, sizeof(path), "/proc/%d/mem", pid);
  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 4;

  uint64_t root = u64(base + rva), context = u64(root + ctx_off);
  uint64_t battle = u64(context + 0x90), world = u64(battle + 0xA8);
  if (!world) { fprintf(stderr, "no live world\n"); return 5; }
  uint64_t player = u64(world + 0xE0 + (uint64_t)which * 8);
  if (!player) { fprintf(stderr, "no player\n"); return 6; }

  static uint8_t object[OBJECT], leaf[LEAF];
  if (!rd(player, object, sizeof(object))) return 7;

  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  int32_t tick = -1;
  rd(battle + 0x60, &tick, 4);

  printf("{\"kind\":\"cr_player_deep\",\"player_index\":%d,\"player\":\"0x%" PRIx64 "\","
         "\"tick\":%d,\"monotonic_us\":%" PRIu64 ",\"object\":\"",
         which, player, tick,
         (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL);
  hex(object, sizeof(object));
  printf("\",\"leaves\":[");

  int first = 1;
  for (size_t off = 0; off + 8 <= sizeof(object); off += 8) {
    uint64_t pointer = 0;
    memcpy(&pointer, object + off, 8);
    if (!plausible(pointer)) continue;
    if (!rd(pointer, leaf, sizeof(leaf))) continue;
    printf("%s{\"off\":\"0x%zx\",\"addr\":\"0x%" PRIx64 "\",\"hex\":\"", first ? "" : ",",
           off, pointer);
    hex(leaf, sizeof(leaf));
    printf("\"}");
    first = 0;
  }
  printf("]}\n");
  close(fd);
  return 0;
}
