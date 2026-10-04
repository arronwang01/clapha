// Read-only: every object in the battle's entity collection, unfiltered, with the global id of
// the data record it points to (object+0x48 -> data+0x40, FirstLight's kObjectDataOffset).
//
//   ent_probe PID MANAGER_RVA CTX_OFF SAMPLES INTERVAL_MS
//
// One JSON line per sample: {"tick":..,"objects":[{"cat":..,"kind":..,"side":..,"x":..,"y":..,
//   "card":..,"data":..,"hp":..},...]}. "card" is what our reader uses today (+0xAC); "data"
// is the object's own data id -- a projectile or area effect names its ProjectileData /
// AreaEffectData there, which is what FirstLight's archetype catalog is keyed by.
#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define UNTAG(p) ((p) & 0x00FFFFFFFFFFFFFFULL)
#define MAX_OBJECTS 512
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

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: ent_probe PID MANAGER_RVA CTX_OFF SAMPLES INTERVAL_MS\n");
    return 2;
  }
  int pid = atoi(argv[1]);
  uint64_t rva = strtoull(argv[2], NULL, 0), ctx_off = strtoull(argv[3], NULL, 0);
  int samples = atoi(argv[4]), interval = atoi(argv[5]);
  uint64_t base = libg_base(pid);
  if (!base) return 3;
  char path[64];
  snprintf(path, sizeof(path), "/proc/%d/mem", pid);
  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 4;
  setvbuf(stdout, NULL, _IONBF, 0);
  for (int sample = 0; sample < samples; ++sample) {
    uint64_t root = 0, context = 0, battle = 0, hp_state = 0, registry = 0, collection = 0, data = 0;
    int32_t tick = -1, count = 0;
    if (!rd(base + rva, &root, 8) || !root || !rd(root + ctx_off, &context, 8) || !context ||
        !rd(context + 0x90, &battle, 8) || !battle || !rd(battle + 0xA8, &hp_state, 8) ||
        !hp_state || !rd(hp_state + 0x08, &registry, 8) || !registry ||
        !rd(registry + 0x40, &collection, 8) || !collection || !rd(collection + 0x08, &data, 8) ||
        !rd(collection + 0x14, &count, 4) || count < 0 || count > MAX_OBJECTS) {
      printf("{\"sample\":%d,\"failure\":\"chain\"}\n", sample);
      usleep((useconds_t)interval * 1000);
      continue;
    }
    rd(battle + 0x60, &tick, 4);
    uint64_t objects[MAX_OBJECTS];
    if (count && (!data || !rd(data, objects, (size_t)count * 8))) count = 0;
    printf("{\"sample\":%d,\"tick\":%d,\"count\":%d,\"objects\":[", sample, tick, count);
    int first = 1;
    for (int i = 0; i < count; ++i) {
      uint8_t raw[0x124];
      if (!objects[i] || !rd(objects[i], raw, sizeof(raw))) continue;
      int32_t category, kind, side, x, y, card, level, data_id = 0, hp = -1;
      uint64_t data_ptr = 0, components = 0, hp_comp = 0;
      memcpy(&category, raw + 0x08, 4);
      memcpy(&kind, raw + 0x30, 4);
      memcpy(&data_ptr, raw + 0x48, 8);
      memcpy(&side, raw + 0x78, 4);
      memcpy(&x, raw + 0x7C, 4);
      memcpy(&y, raw + 0x80, 4);
      memcpy(&card, raw + 0xAC, 4);
      memcpy(&level, raw + 0x120, 4);
      memcpy(&components, raw + 0x18, 8);
      if (data_ptr) rd(data_ptr + 0x40, &data_id, 4);
      if (components && rd(components + 0x10, &hp_comp, 8) && hp_comp) rd(hp_comp + 0x10, &hp, 4);
      printf("%s{\"cat\":%d,\"kind\":%d,\"side\":%d,\"x\":%d,\"y\":%d,\"card\":%d,\"data\":%u,"
             "\"hp\":%d}", first ? "" : ",", category, kind, side, x, y, card, (uint32_t)data_id, hp);
      first = 0;
    }
    printf("]}\n");
    usleep((useconds_t)interval * 1000);
  }
  close(fd);
  return 0;
}
