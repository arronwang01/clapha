// Locate hero/champion ability controllers and evolution progress in a live battle.
//
// FirstLight's in-process probe (Null's 15.535.13) reads, per player:
//   player + 0x3a0 + i*8  -> ability controller (i = 0, 1)
//       controller + 0x20  == player (back-pointer)      <- the signature searched for here
//       controller + 0x18  -> action data (+0x40 global id)
//       controller + 0x78  remaining cooldown ms, + 0x7c configured cooldown ms
//       controller + 0x80  remaining charges (-1 unlimited), + 0x98 button state
//       controller + 0x90  -> selected hero/champion character data (+0x40 global id)
//   player + 0x2e8          native vector, one int32 evolution progress per deck slot
// This build moved some player fields by 0x10 (hand 0x220 -> 0x210) but not others (elixir
// stays 0x2f8), so nothing is assumed: controllers are found by their back-pointer anywhere in
// player+0x280..0x480, and progress vectors as any 8-slot int vector in player+0x240..0x340.
//
// Read-only (/proc/PID/mem). Prints one JSON line per sample.
// usage: runtime_probe PID MANAGER_RVA CTX_OFF SAMPLES INTERVAL_MS
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
  uint64_t best = UINT64_MAX;
  snprintf(path, sizeof(path), "/proc/%d/maps", pid);
  FILE *maps = fopen(path, "r");
  if (!maps) return 0;
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

static void controllers(uint64_t player) {
  int first = 1;
  printf("\"controllers\":[");
  for (uint32_t off = 0x280; off < 0x480; off += 8) {
    uint64_t controller = 0, back = 0, action = 0, character = 0;
    if (!rd(player + off, &controller, 8) || !controller) continue;
    if (!rd(controller + 0x20, &back, 8) || UNTAG(back) != UNTAG(player)) continue;
    int32_t cooldown = 0, configured = 0, charges = 0, button = 0, action_id = 0, character_id = 0;
    rd(controller + 0x78, &cooldown, 4);
    rd(controller + 0x7c, &configured, 4);
    rd(controller + 0x80, &charges, 4);
    rd(controller + 0x98, &button, 4);
    if (rd(controller + 0x18, &action, 8) && action) rd(action + 0x40, &action_id, 4);
    if (rd(controller + 0x90, &character, 8) && character) rd(character + 0x40, &character_id, 4);
    printf("%s{\"player_off\":\"0x%x\",\"cooldown_ms\":%d,\"configured_ms\":%d,\"charges\":%d,"
           "\"button\":%d,\"action_id\":%d,\"character_id\":%d}",
           first ? "" : ",", off, cooldown, configured, charges, button, action_id, character_id);
    first = 0;
  }
  printf("]");
}

static void vectors(uint64_t player) {
  int first = 1;
  printf(",\"vectors8\":[");
  for (uint32_t off = 0x240; off < 0x340; off += 8) {
    uint64_t data = 0;
    int32_t capacity = -1, count = -1, values[8];
    if (!rd(player + off, &data, 8) || !data) continue;
    if (!rd(player + off + 0x08, &capacity, 4) || !rd(player + off + 0x0c, &count, 4)) continue;
    if (count != 8 || capacity < 8 || capacity > 16) continue;
    if (!rd(data, values, sizeof(values))) continue;
    int small = 1;
    for (int i = 0; i < 8; ++i)
      if (values[i] < -1 || values[i] > 64) small = 0;
    if (!small) continue;
    printf("%s{\"player_off\":\"0x%x\",\"values\":[%d,%d,%d,%d,%d,%d,%d,%d]}", first ? "" : ",",
           off, values[0], values[1], values[2], values[3], values[4], values[5], values[6],
           values[7]);
    first = 0;
  }
  printf("]");
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: runtime_probe PID MANAGER_RVA CTX_OFF SAMPLES INTERVAL_MS\n");
    return 2;
  }
  int pid = atoi(argv[1]);
  uint64_t rva = strtoull(argv[2], NULL, 0), ctx_off = strtoull(argv[3], NULL, 0);
  int samples = atoi(argv[4]), interval_ms = atoi(argv[5]);
  uint64_t base = libg_base(pid);
  if (!base) { fprintf(stderr, "libg not mapped\n"); return 3; }
  char path[64];
  snprintf(path, sizeof(path), "/proc/%d/mem", pid);
  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) { perror("mem"); return 4; }
  setvbuf(stdout, NULL, _IONBF, 0);
  for (int sample = 0; samples <= 0 || sample < samples; ++sample) {
    uint64_t root = 0, context = 0, battle = 0, world = 0;
    int32_t tick = -1;
    if (!rd(base + rva, &root, 8) || !root || !rd(root + ctx_off, &context, 8) || !context ||
        !rd(context + 0x90, &battle, 8) || !battle || !rd(battle + 0xA8, &world, 8) || !world) {
      printf("{\"sample\":%d,\"failure\":\"chain\"}\n", sample);
      usleep((useconds_t)interval_ms * 1000);
      continue;
    }
    rd(battle + 0x60, &tick, 4);
    printf("{\"sample\":%d,\"tick\":%d,\"players\":[", sample, tick);
    for (int side = 0; side < 2; ++side) {
      uint64_t player = 0;
      int32_t elixir = -1;
      if (side) putchar(',');
      if (!rd(world + 0xE0 + (uint64_t)side * 8, &player, 8) || !player) {
        printf("null");
        continue;
      }
      rd(player + 0x2F8, &elixir, 4);
      printf("{\"slot\":%d,\"elixir_raw\":%d,", side, elixir);
      controllers(player);
      vectors(player);
      printf("}");
    }
    printf("]}\n");
    usleep((useconds_t)interval_ms * 1000);
  }
  close(fd);
  return 0;
}
