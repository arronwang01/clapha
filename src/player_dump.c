// Raw dump of both player objects, to settle where the opponent's hand/cycle live.
//
// Upstream (002) reads: hand vec +0x210 (cap +0x218, len +0x21c),
//                       cycle vec +0x220 (cap +0x228, len +0x22c), elixir +0x2f8.
// The user's Null's 15.535.13 notes say: hand +0x220, cycle +0x230, elixir +0x2f8,
// revealed cards +0x288..+0x2a4, and local account id at world+0x2e4.
// Those differ by 0x10, so dump the region and decide from the data.
//
// usage: player_dump PID MANAGER_RVA CTX_OFF
#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define UNTAG(p) ((p) & 0x00FFFFFFFFFFFFFFULL)
static int fd;

static int rd(uint64_t a, void *o, size_t n) {
  return pread(fd, o, n, (off_t)UNTAG(a)) == (ssize_t)n;
}
static uint64_t u64(uint64_t a) { uint64_t v = 0; rd(a, &v, 8); return v; }
static int32_t i32(uint64_t a) { int32_t v = -1; rd(a, &v, 4); return v; }

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

/* A candidate int32 vector: {data ptr, capacity, length} with small deck indices. */
static void scan_vectors(uint64_t player, int from, int to) {
  printf("      \"vector_candidates\":[");
  int first = 1;
  for (int off = from; off <= to; off += 8) {
    uint64_t data = u64(player + off);
    int32_t capacity = i32(player + off + 8), length = i32(player + off + 12);
    if (!data || capacity < 1 || capacity > 16 || length < 0 || length > capacity) continue;
    int32_t values[16];
    int ok = 1;
    for (int i = 0; i < length && ok; ++i) {
      values[i] = i32(UNTAG(data) + (uint64_t)i * 4);
      if (values[i] < -1 || values[i] > 7) ok = 0;
    }
    if (!ok) continue;
    printf("%s{\"off\":\"0x%x\",\"cap\":%d,\"len\":%d,\"values\":[", first ? "" : ",",
           off, capacity, length);
    for (int i = 0; i < length; ++i) printf("%s%d", i ? "," : "", values[i]);
    printf("]}");
    first = 0;
  }
  printf("],\n");
}

int main(int argc, char **argv) {
  if (argc != 4) { fprintf(stderr, "usage: player_dump PID MANAGER_RVA CTX_OFF\n"); return 2; }
  int pid = atoi(argv[1]);
  uint64_t rva = strtoull(argv[2], NULL, 0), ctx_off = strtoull(argv[3], NULL, 0);
  uint64_t base = libg_base(pid);
  if (!base) return 3;
  char path[64];
  snprintf(path, sizeof(path), "/proc/%d/mem", pid);
  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 4;

  uint64_t root = u64(base + rva), context = u64(root + ctx_off);
  uint64_t battle = u64(context + 0x90), world = u64(battle + 0xA8);
  printf("{\n  \"world\":\"0x%" PRIx64 "\",\"world_0x2e4\":%d,\"world_0x2e0\":%d,\n",
         world, i32(world + 0x2e4), i32(world + 0x2e0));
  printf("  \"players\":[\n");
  for (int index = 0; index < 2; ++index) {
    uint64_t player = u64(world + 0xE0 + (uint64_t)index * 8);
    printf("    {\"index\":%d,\"player\":\"0x%" PRIx64 "\",\"side_0x78\":%d,\"elixir_0x2f8\":%d,\n",
           index, player, i32(player + 0x78), i32(player + 0x2f8));
    scan_vectors(player, 0x200, 0x260);
    printf("      \"revealed_0x288_0x2a8\":[");
    for (int off = 0x288; off < 0x2a8; off += 4)
      printf("%s%d", off == 0x288 ? "" : ",", i32(player + off));
    printf("],\n      \"words_0x200_0x300\":[");
    for (int off = 0x200; off < 0x300; off += 4)
      printf("%s[\"0x%x\",%d]", off == 0x200 ? "" : ",", off, i32(player + off));
    printf("]}%s\n", index == 0 ? "," : "");
  }
  printf("  ]\n}\n");
  close(fd);
  return 0;
}
