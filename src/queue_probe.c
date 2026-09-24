// Read-only probe for two structures described in the user's prior notes, tested here
// against build 160402012. Their "manager" is this chain's *battle* object
// (their world at manager+0xa8 == our player_state at battle+0xA8).
//
//   battle+0x60  simulated tick (already confirmed by us)
//   battle+0x64  newest tick received from the server  (expect a gap of ~11..32)
//   battle+0x38  command queue: vector data at +0x08, count at +0x14
//     entry +0x10 issue tick, +0x18 account id low32, +0x28/+0x2c x/y,
//           +0x38 card LogicData (card id at +0x40), +0x50 sequence
//
// Prints a raw window around the battle header too, so the fields can be identified
// if the offsets moved. Reads only; sends nothing.
//
// usage: queue_probe PID MANAGER_RVA CTX_OFF SAMPLES INTERVAL_MS
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

static int rd(uint64_t address, void *out, size_t size) {
  return pread(fd, out, size, (off_t)UNTAG(address)) == (ssize_t)size;
}

static uint64_t libg_base(int pid) {
  char path[64], line[1024];
  snprintf(path, sizeof(path), "/proc/%d/maps", pid);
  FILE *maps = fopen(path, "r");
  if (!maps) return 0;
  uint64_t best = UINT64_MAX;
  while (fgets(line, sizeof(line), maps)) {
    unsigned long long start = 0, offset = 0;
    char perms[8] = {0};
    if (!strstr(line, "/libg.so")) continue;
    if (sscanf(line, "%llx-%*llx %7s %llx", &start, perms, &offset) != 3) continue;
    if (start >= offset && start - offset < best) best = start - offset;
  }
  fclose(maps);
  return best == UINT64_MAX ? 0 : best;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: queue_probe PID MANAGER_RVA CTX_OFF SAMPLES INTERVAL_MS\n");
    return 2;
  }
  int pid = atoi(argv[1]);
  uint64_t rva = strtoull(argv[2], NULL, 0), ctx_off = strtoull(argv[3], NULL, 0);
  int samples = atoi(argv[4]), interval_ms = atoi(argv[5]);
  if (pid <= 0 || samples < 0 || samples > 200000 || interval_ms < 20) return 2;
  if (samples == 0) samples = 200000;  /* 0 = run until killed */

  uint64_t base = libg_base(pid);
  if (!base) { fprintf(stderr, "libg not mapped\n"); return 3; }
  char path[64];
  snprintf(path, sizeof(path), "/proc/%d/mem", pid);
  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) { perror("mem"); return 4; }
  setvbuf(stdout, NULL, _IONBF, 0);

  for (int sample = 0; sample < samples; ++sample) {
    uint64_t root = 0, context = 0, battle = 0;
    int32_t simulated = -1, received = -1;
    if (!rd(base + rva, &root, 8) || !root ||
        !rd(root + ctx_off, &context, 8) || !context ||
        !rd(context + 0x90, &battle, 8) || !battle) {
      printf("{\"sample\":%d,\"failure\":\"chain\"}\n", sample);
      usleep((useconds_t)interval_ms * 1000);
      continue;
    }
    rd(battle + 0x60, &simulated, 4);
    rd(battle + 0x64, &received, 4);

    printf("{\"sample\":%d,\"battle\":\"0x%" PRIx64 "\",\"tick_0x60\":%d,\"field_0x64\":%d,"
           "\"gap\":%d,\"header\":[",
           sample, battle, simulated, received, received - simulated);
    for (uint32_t off = 0x30; off < 0x80; off += 4) {
      int32_t value = 0;
      rd(battle + off, &value, 4);
      printf("%s[\"0x%x\",%d]", off == 0x30 ? "" : ",", off, value);
    }
    printf("],\"revealed\":[");
    {
      uint64_t world = 0;
      rd(battle + 0xA8, &world, 8);
      for (int side = 0; side < 2; ++side) {
        uint64_t player = 0;
        if (side) putchar(',');
        printf("[");
        if (world && rd(UNTAG(world) + 0xE0 + (uint64_t)side * 8, &player, 8) && player) {
          int first_card = 1;
          for (uint32_t off = 0x288; off < 0x2a8; off += 4) {
            int32_t card = -1;
            rd(player + off, &card, 4);
            if (card <= 0) continue;
            printf("%s%d", first_card ? "" : ",", card);
            first_card = 0;
          }
        }
        printf("]");
      }
    }
    printf("],\"accounts\":[");
    {
      uint64_t world = 0;
      rd(battle + 0xA8, &world, 8);
      for (int side = 0; side < 2; ++side) {
        uint64_t player = 0, pctx = 0, proot = 0, avatar = 0;
        int32_t side_field = -1, hi = 0, lo = 0;
        if (side) putchar(',');
        if (world && rd(UNTAG(world) + 0xE0 + (uint64_t)side * 8, &player, 8) && player &&
            rd(player + 0x10, &pctx, 8) && pctx && rd(UNTAG(pctx) + 0x98, &proot, 8) && proot &&
            rd(player + 0x78, &side_field, 4) && side_field >= 0 && side_field < 6 &&
            rd(UNTAG(proot) + 0x30 + (uint64_t)side_field * 8, &avatar, 8) && avatar) {
          rd(UNTAG(avatar), &hi, 4);
          rd(UNTAG(avatar) + 4, &lo, 4);
          printf("{\"side\":%d,\"hi\":%d,\"lo\":%d}", side_field, hi, lo);
        } else {
          printf("null");
        }
      }
    }
    printf("],\"queue\":");

    uint64_t queue = 0, data = 0;
    int32_t count = -1;
    if (rd(battle + 0x38, &queue, 8) && queue &&
        rd(UNTAG(queue) + 0x08, &data, 8) && rd(UNTAG(queue) + 0x14, &count, 4) &&
        count >= 0 && count < 64 && data) {
      printf("{\"object\":\"0x%" PRIx64 "\",\"count\":%d,\"entries\":[", queue, count);
      for (int index = 0; index < count; ++index) {
        uint64_t entry = 0, logic = 0;
        int32_t issue_tick = -1, account = 0, x = -1, y = -1, card = -1, sequence = -1;
        if (!rd(UNTAG(data) + (uint64_t)index * 8, &entry, 8) || !entry) continue;
        rd(entry + 0x10, &issue_tick, 4);
        rd(entry + 0x18, &account, 4);
        rd(entry + 0x28, &x, 4);
        rd(entry + 0x2c, &y, 4);
        rd(entry + 0x50, &sequence, 4);
        if (rd(entry + 0x38, &logic, 8) && logic) rd(UNTAG(logic) + 0x40, &card, 4);
        printf("%s{\"addr\":\"0x%" PRIx64 "\",\"issue_tick\":%d,\"account_lo\":%d,"
               "\"x\":%d,\"y\":%d,\"card_id\":%d,\"seq\":%d}",
               index ? "," : "", entry, issue_tick, account, x, y, card, sequence);
      }
      printf("]}");
    } else {
      printf("{\"object\":\"0x%" PRIx64 "\",\"count\":%d,\"plausible\":false}", queue, count);
    }
    printf("}\n");
    usleep((useconds_t)interval_ms * 1000);
  }
  close(fd);
  return 0;
}
