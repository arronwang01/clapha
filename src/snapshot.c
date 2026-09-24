// Gentle bulk snapshot: resolve the chain once, then copy a few whole regions in single
// pread calls and stop touching the process. Everything else is analysed offline on the copy.
//
// This is the opposite of chasing pointers every tick: 1 chain walk + ~8 block reads,
// instead of hundreds of small reads per second.
//
// Output is one JSON object with hex blocks, so it can be shared and analysed anywhere.
//
// usage: snapshot PID MANAGER_RVA CTX_OFF
#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define UNTAG(p) ((p) & 0x00FFFFFFFFFFFFFFULL)
#define BLOCK 0x600

static int fd;

static int rd(uint64_t a, void *o, size_t n) {
  return pread(fd, o, n, (off_t)UNTAG(a)) == (ssize_t)n;
}
static uint64_t u64(uint64_t a) { uint64_t v = 0; rd(a, &v, 8); return v; }

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

static void emit_block(const char *name, uint64_t address, size_t size, int *first) {
  static uint8_t buffer[0x4000];
  if (size > sizeof(buffer)) size = sizeof(buffer);
  if (!address || !rd(address, buffer, size)) return;
  printf("%s\n    {\"name\":\"%s\",\"address\":\"0x%" PRIx64 "\",\"size\":%zu,\"hex\":\"",
         *first ? "" : ",", name, address, size);
  for (size_t i = 0; i < size; ++i) printf("%02x", buffer[i]);
  printf("\"}");
  *first = 0;
}

int main(int argc, char **argv) {
  if (argc != 4) { fprintf(stderr, "usage: snapshot PID MANAGER_RVA CTX_OFF\n"); return 2; }
  int pid = atoi(argv[1]);
  uint64_t rva = strtoull(argv[2], NULL, 0), ctx_off = strtoull(argv[3], NULL, 0);
  uint64_t base = libg_base(pid);
  if (!base) return 3;
  char path[64];
  snprintf(path, sizeof(path), "/proc/%d/mem", pid);
  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 4;

  /* one chain walk */
  uint64_t root = u64(base + rva), context = u64(root + ctx_off);
  uint64_t battle = u64(context + 0x90), world = u64(battle + 0xA8);
  uint64_t player0 = u64(world + 0xE0), player1 = u64(world + 0xE8);
  uint64_t pcontext = u64(player1 + 0x10), proot = u64(pcontext + 0x98);
  uint64_t owner0 = u64(proot + 0x88), owner1 = u64(proot + 0x90);
  uint64_t entries0 = owner0 ? u64(owner0 + 0x20) : 0;
  uint64_t entries1 = owner1 ? u64(owner1 + 0x20) : 0;
  uint64_t avatar0 = u64(proot + 0x30), avatar1 = u64(proot + 0x38);
  /* hand/cycle vector payloads for both players (the arrays the headers point at) */
  uint64_t hand0 = u64(player0 + 0x210), cycle0 = u64(player0 + 0x220);
  uint64_t hand1 = u64(player1 + 0x210), cycle1 = u64(player1 + 0x220);

  printf("{\n  \"kind\":\"cr_live_snapshot\",\"version_code\":160402012,\"pid\":%d,\n"
         "  \"chain\":{\"libg_base\":\"0x%" PRIx64 "\",\"manager_rva\":\"0x%" PRIx64 "\","
         "\"root\":\"0x%" PRIx64 "\",\"context\":\"0x%" PRIx64 "\",\"battle\":\"0x%" PRIx64 "\","
         "\"world\":\"0x%" PRIx64 "\",\"player0\":\"0x%" PRIx64 "\",\"player1\":\"0x%" PRIx64 "\","
         "\"player_root\":\"0x%" PRIx64 "\",\"owner0\":\"0x%" PRIx64 "\",\"owner1\":\"0x%" PRIx64 "\"},\n"
         "  \"blocks\":[",
         pid, base, rva, root, context, battle, world, player0, player1, proot, owner0, owner1);

  int first = 1;
  emit_block("battle", battle, BLOCK, &first);
  emit_block("world", world, 0x800, &first);
  emit_block("player0", player0, BLOCK, &first);
  emit_block("player1", player1, BLOCK, &first);
  emit_block("player_root", proot, BLOCK, &first);
  emit_block("owner0", owner0, 0x100, &first);
  emit_block("owner1", owner1, 0x100, &first);
  emit_block("owner0_entries", entries0, 0x80, &first);
  emit_block("owner1_entries", entries1, 0x80, &first);
  emit_block("avatar0", avatar0, 0x200, &first);
  emit_block("avatar1", avatar1, 0x200, &first);
  emit_block("player0_hand_data", hand0, 0x40, &first);
  emit_block("player0_cycle_data", cycle0, 0x40, &first);
  emit_block("player1_hand_data", hand1, 0x40, &first);
  emit_block("player1_cycle_data", cycle1, 0x40, &first);
  printf("\n  ]\n}\n");
  close(fd);
  return 0;
}
