// Deck scan, following the repo's scanner pattern (mumu_arm64_tick_scan.c /
// mumu_cycle_vector_scan.c) rather than a general heap sweep:
//
//   * only scudo: heaps and low anonymous native maps, never dalvik spaces
//   * a candidate is only reported if it sits inside a real game object, proven by looking
//     back up to 0x300 (8-byte aligned) for a first qword that is a vtable pointer inside
//     libg's mapped range
//   * the deck shape is also checked as a vector, the way mumu_cycle_vector_scan.c does:
//     {data ptr, capacity int32, size int32} with a plausible capacity
//
// Needle: eight distinct card ids, table*1000000 + small row, not a sequential catalogue run.
//
// usage: deck_scan PID
#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_MAPS 2048
#define MAX_HITS 2000
#define CHUNK (64 * 1024)
#define DECK 8

typedef struct {
  uint64_t start, end;
  int readable, writable;
  char path[512];
} MapRange;

static int fd;

static int read_exact(uint64_t address, void *output, size_t size) {
  uint8_t *cursor = output;
  size_t done = 0;
  while (done < size) {
    ssize_t value = pread(fd, cursor + done, size - done,
                          (off_t)((address & 0x00FFFFFFFFFFFFFFULL) + done));
    if (value <= 0) return 0;
    done += (size_t)value;
  }
  return 1;
}

static int parse_maps(int pid, MapRange maps[MAX_MAPS]) {
  char filename[64], line[2048];
  snprintf(filename, sizeof(filename), "/proc/%d/maps", pid);
  FILE *handle = fopen(filename, "r");
  if (!handle) return -1;
  int count = 0;
  while (count < MAX_MAPS && fgets(line, sizeof(line), handle)) {
    unsigned long long start = 0, end = 0, offset = 0;
    unsigned int major = 0, minor = 0;
    unsigned long inode = 0;
    char permissions[8] = {0};
    int consumed = 0;
    if (sscanf(line, "%llx-%llx %7s %llx %x:%x %lu %n", &start, &end, permissions,
               &offset, &major, &minor, &inode, &consumed) < 7)
      continue;
    MapRange *out = &maps[count++];
    memset(out, 0, sizeof(*out));
    out->start = start;
    out->end = end;
    out->readable = permissions[0] == 'r';
    out->writable = permissions[1] == 'w';
    char *name = line + consumed;
    while (*name == ' ' || *name == '\t') ++name;
    size_t length = strcspn(name, "\r\n");
    if (length >= sizeof(out->path)) length = sizeof(out->path) - 1;
    memcpy(out->path, name, length);
  }
  fclose(handle);
  return count;
}

static int is_card(int32_t value) {
  int table = value / 1000000, row = value % 1000000;
  if (row < 0 || row > 300) return 0;
  return table == 26 || table == 27 || table == 28 || table == 13 || table == 203;
}

static int deck_like(const int32_t *values) {
  for (int i = 0; i < DECK; ++i) {
    if (!is_card(values[i])) return 0;
    for (int k = i + 1; k < DECK; ++k)
      if (values[i] == values[k]) return 0;
  }
  int run = 1;                          /* a sequential block is the catalogue, not a deck */
  for (int i = 1; i < DECK; ++i)
    if (values[i] != values[i - 1] + 1) { run = 0; break; }
  return !run;
}

int main(int argc, char **argv) {
  if (argc != 2) { fprintf(stderr, "usage: deck_scan PID\n"); return 2; }
  int pid = atoi(argv[1]);
  MapRange maps[MAX_MAPS];
  int map_count = parse_maps(pid, maps);
  if (map_count <= 0) return 3;

  uint64_t libg_min = UINT64_MAX, libg_max = 0;
  for (int index = 0; index < map_count; ++index) {
    if (strstr(maps[index].path, "/libg.so")) {
      if (maps[index].start < libg_min) libg_min = maps[index].start;
      if (maps[index].end > libg_max) libg_max = maps[index].end;
    }
  }
  char memory_path[64];
  snprintf(memory_path, sizeof(memory_path), "/proc/%d/mem", pid);
  fd = open(memory_path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 4;

  printf("{\"pid\":%d,\"libg\":[\"0x%" PRIx64 "\",\"0x%" PRIx64 "\"],\"hits\":[",
         pid, libg_min, libg_max);
  int first = 1, hits = 0;
  uint64_t scanned = 0;
  uint8_t *buffer = malloc(CHUNK);
  if (!buffer) return 5;

  for (int map_index = 0; map_index < map_count && hits < MAX_HITS; ++map_index) {
    MapRange *map = &maps[map_index];
    /* the repo's map filter, verbatim in intent: scudo heaps and low native maps only */
    int scudo = strstr(map->path, "scudo:") != NULL;
    int low_native = map->end < 0x100000000ULL &&
        (map->path[0] == 0 || strstr(map->path, "Mem_"));
    if (!map->readable || !map->writable || (!scudo && !low_native) ||
        map->end <= map->start)
      continue;

    for (uint64_t start = map->start; start < map->end && hits < MAX_HITS; start += CHUNK) {
      size_t size = (size_t)((map->end - start) < CHUNK ? (map->end - start) : CHUNK);
      if (!read_exact(start, buffer, size)) continue;
      scanned += size;
      size_t words = size / 4;
      const int32_t *value = (const int32_t *)buffer;

      /* A deck is not an array of ints: it is eight POINTERS to entry objects, where
         entry+0x10 -> data and data+0x40 is the card id (the shape read_visible_deck walks).
         So the needle is eight consecutive qwords that all resolve that way. */
      const uint64_t *slots = (const uint64_t *)buffer;
      size_t qwords = size / 8;
      for (size_t q = 0; q + DECK <= qwords && hits < MAX_HITS; ++q) {
        int32_t resolved[DECK];
        int ok = 1;
        for (int k = 0; k < DECK && ok; ++k) {
          uint64_t entry = slots[q + k] & 0x00FFFFFFFFFFFFFFULL, data = 0;
          if (entry < 0x10000) { ok = 0; break; }
          if (!read_exact(entry + 0x10, &data, 8) || (data & 0x00FFFFFFFFFFFFFFULL) < 0x10000) {
            ok = 0; break;
          }
          if (!read_exact((data & 0x00FFFFFFFFFFFFFFULL) + 0x40, &resolved[k], 4)) { ok = 0; break; }
          if (!is_card(resolved[k])) ok = 0;
        }
        if (!ok || !deck_like(resolved)) continue;
        uint64_t address = start + (uint64_t)q * 8;
        printf("%s{\"entries\":\"0x%" PRIx64 "\",\"map\":\"%s\",\"cards\":[",
               first ? "" : ",", address, map->path);
        for (int k = 0; k < DECK; ++k) printf("%s%d", k ? "," : "", resolved[k]);
        printf("]}");
        first = 0;
        ++hits;
      }

      for (size_t i = 0; i + DECK <= words && hits < MAX_HITS; ++i) {
        if (!is_card(value[i]) || !deck_like(&value[i])) continue;
        uint64_t address = start + (uint64_t)i * 4;

        /* the repo's object test: look back for a vtable pointing into libg */
        uint64_t object = 0, vtable = 0;
        int field_offset = -1;
        for (int back = 0; back <= 0x300; back += 8) {
          if (address < (uint64_t)back) break;
          uint64_t candidate = address - (uint64_t)back;
          if ((candidate & 7) != 0 || candidate < start) continue;
          uint64_t maybe = 0;
          memcpy(&maybe, buffer + (candidate - start), 8);
          if (maybe < libg_min || maybe >= libg_max) continue;
          object = candidate;
          vtable = maybe;
          field_offset = back;
          break;
        }

        /* and the vector test from mumu_cycle_vector_scan.c: is something pointing at this
           array as {data, capacity, size}? reported when the header sits just before it */
        int32_t capacity = -1, length = -1;
        if (address >= start + 16) {
          uint64_t header = address - 16;
          uint64_t data = 0;
          memcpy(&data, buffer + (header - start), 8);
          memcpy(&capacity, buffer + (header - start) + 8, 4);
          memcpy(&length, buffer + (header - start) + 12, 4);
          if ((data & 0x00FFFFFFFFFFFFFFULL) != (address & 0x00FFFFFFFFFFFFFFULL)) {
            capacity = length = -1;
          }
        }

        printf("%s{\"address\":\"0x%" PRIx64 "\",\"map\":\"%s\",\"object\":\"0x%" PRIx64
               "\",\"vtable\":\"0x%" PRIx64 "\",\"field_offset\":%d,"
               "\"vector_capacity\":%d,\"vector_length\":%d,\"cards\":[",
               first ? "" : ",", address, map->path, object, vtable, field_offset,
               capacity, length);
        for (int k = 0; k < DECK; ++k) printf("%s%d", k ? "," : "", value[i + k]);
        printf("]}");
        first = 0;
        ++hits;
        i += DECK - 1;
      }
    }
  }
  printf("],\"hit_count\":%d,\"bytes_scanned\":%" PRIu64 "}\n", hits, scanned);
  free(buffer);
  close(fd);
  return 0;
}
