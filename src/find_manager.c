// Read-only finder for the battle manager global in libg.so.
// Walks every 8-byte slot in libg's writable segments (and the .bss right after),
// follows slot -> root -> +ctx_off -> context -> +0x90 -> battle -> +0x60 tick,
// and reports slots whose tick advances at ~20 Hz between two samples.
// Usage: find_manager PID [CTX_OFF_MIN CTX_OFF_MAX]   (default ctx offset 0x28 only)
#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_MAPS 4096
#define MAX_CANDIDATES 200000

typedef struct { uint64_t start, end; char perms[8]; char path[256]; } Map;
typedef struct { uint64_t slot, root, context, battle; uint32_t ctx_off; int32_t tick0; } Cand;

static int fd;

// Android heap pointers carry a tag in the top byte (e.g. 0xb4...); strip it before use.
static uint64_t untag(uint64_t p) { return p & 0x00FFFFFFFFFFFFFFULL; }

static int rd(uint64_t address, void *out, size_t size) {
  return pread(fd, out, size, (off_t)untag(address)) == (ssize_t)size;
}

static int plausible_ptr(uint64_t p) {
  p = untag(p);
  return p >= 0x10000 && p < 0x0000800000000000ULL && (p & 7) == 0;
}

static uint64_t now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 4) {
    fprintf(stderr, "usage: find_manager PID [CTX_OFF_MIN CTX_OFF_MAX]\n");
    return 2;
  }
  int pid = atoi(argv[1]);
  uint32_t ctx_min = 0x28, ctx_max = 0x28;
  if (argc == 4) {
    ctx_min = (uint32_t)strtoul(argv[2], NULL, 0);
    ctx_max = (uint32_t)strtoul(argv[3], NULL, 0);
  }

  char path[64];
  snprintf(path, sizeof(path), "/proc/%d/maps", pid);
  FILE *maps_file = fopen(path, "r");
  if (!maps_file) { perror("maps"); return 3; }
  static Map maps[MAX_MAPS];
  int map_count = 0;
  char line[1024];
  uint64_t libg_base = UINT64_MAX, libg_end = 0;
  while (map_count < MAX_MAPS && fgets(line, sizeof(line), maps_file)) {
    unsigned long long start, end, offset;
    Map *m = &maps[map_count];
    int consumed = 0;
    memset(m, 0, sizeof(*m));
    if (sscanf(line, "%llx-%llx %7s %llx %*s %*s %n", &start, &end, m->perms,
               &offset, &consumed) < 4)
      continue;
    m->start = start;
    m->end = end;
    if (consumed) {
      char *name = line + consumed;
      size_t length = strcspn(name, "\r\n");
      if (length >= sizeof(m->path)) length = sizeof(m->path) - 1;
      memcpy(m->path, name, length);
    }
    if (strstr(m->path, "/libg.so")) {
      if (start - offset < libg_base) libg_base = start - offset;
      if (end > libg_end) libg_end = end;
    }
    ++map_count;
  }
  fclose(maps_file);
  if (libg_base == UINT64_MAX) { fprintf(stderr, "libg.so not mapped\n"); return 3; }

  snprintf(path, sizeof(path), "/proc/%d/mem", pid);
  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) { perror("mem"); return 4; }

  static Cand cands[MAX_CANDIDATES];
  int cand_count = 0;
  uint64_t slots_scanned = 0;
  // libg's own writable segments plus the anonymous .bss that follows it (within 64 MB).
  for (int i = 0; i < map_count; ++i) {
    Map *m = &maps[i];
    if (m->perms[0] != 'r' || m->perms[1] != 'w') continue;
    int is_libg = strstr(m->path, "/libg.so") != NULL;
    // .bss can sit between libg's file-backed segments, so accept anonymous rw maps
    // anywhere from the libg base up to 64 MB past its last segment.
    int is_bss = (m->path[0] == 0 || strstr(m->path, ".bss")) &&
                 m->start >= libg_base && m->start < libg_end + 0x4000000ULL;
    if (!is_libg && !is_bss) continue;
    size_t size = (size_t)(m->end - m->start);
    uint64_t *buffer = malloc(size);
    if (!buffer || !rd(m->start, buffer, size)) { free(buffer); continue; }
    for (size_t k = 0; k < size / 8; ++k) {
      ++slots_scanned;
      uint64_t root = buffer[k];
      if (!plausible_ptr(root)) continue;
      for (uint32_t off = ctx_min; off <= ctx_max; off += 8) {
        uint64_t context = 0, battle = 0;
        int32_t tick = -1;
        if (!rd(root + off, &context, 8) || !plausible_ptr(context)) continue;
        if (!rd(context + 0x90, &battle, 8) || !plausible_ptr(battle)) continue;
        if (!rd(battle + 0x60, &tick, 4) || tick < 0 || tick > 20 * 60 * 30) continue;
        if (cand_count < MAX_CANDIDATES)
          cands[cand_count++] = (Cand){m->start + k * 8, root, context, battle, off, tick};
      }
    }
    free(buffer);
  }

  uint64_t t0 = now_us();
  usleep(1000000);
  double elapsed = (now_us() - t0) / 1e6;

  printf("{\"libg_base\":\"0x%" PRIx64 "\",\"slots_scanned\":%" PRIu64
         ",\"chain_shaped\":%d,\"elapsed_s\":%.3f,\"hits\":[",
         libg_base, slots_scanned, cand_count, elapsed);
  int hits = 0;
  for (int i = 0; i < cand_count; ++i) {
    Cand *c = &cands[i];
    uint64_t root = 0, context = 0, battle = 0;
    int32_t tick = -1;
    if (!rd(c->slot, &root, 8) || root != c->root) continue;
    if (!rd(root + c->ctx_off, &context, 8) || context != c->context) continue;
    if (!rd(context + 0x90, &battle, 8) || battle != c->battle) continue;
    if (!rd(battle + 0x60, &tick, 4)) continue;
    double rate = (tick - c->tick0) / elapsed;
    if (tick == c->tick0) continue;  // report any advancing tick; rate checked by eye
    printf("%s{\"rva\":\"0x%" PRIx64 "\",\"ctx_off\":\"0x%x\",\"root\":\"0x%" PRIx64
           "\",\"battle\":\"0x%" PRIx64 "\",\"tick_before\":%d,\"tick_after\":%d,\"rate\":%.1f}",
           hits++ ? "," : "", c->slot - libg_base, c->ctx_off, root, battle, c->tick0, tick, rate);
  }
  printf("],\"hit_count\":%d}\n", hits);
  close(fd);
  return 0;
}
