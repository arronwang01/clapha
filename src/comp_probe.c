// Read-only component probe: for each given troop object, list its components and the
// candidate attack/charge fields named by FirstLight's probe layout headers.
//
//   comp_probe PID SAMPLES INTERVAL_MS ADDR [ADDR...]
//
// Per sample and object it prints one JSON line:
//   {"t":ms,"obj":"0x..","x":..,"y":..,"state":..,"deploy":..,
//    "comps":[{"i":0,"vt":"0xRVA","owner":1,"p10":"0x..","w20":..,"w24":..,"w28":..,"w1e0":..},...]}
// vt is the component's vtable as an offset into libg, so components of the same type share it
// across objects. owner=1 when component+0x08 points back at the object (their
// kComponentOwnerOffset). Nothing is written, nothing is hooked; pread on /proc/PID/mem only.
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
#define MAX_COMPONENTS 16

static int fd;

static int rd(uint64_t address, void *out, size_t size) {
  return pread(fd, out, size, (off_t)UNTAG(address)) == (ssize_t)size;
}

static uint64_t libg_base(int pid) {
  char path[64], line[512];
  snprintf(path, sizeof(path), "/proc/%d/maps", pid);
  FILE *maps = fopen(path, "r");
  if (!maps) return 0;
  uint64_t base = 0;
  while (fgets(line, sizeof(line), maps)) {
    if (!strstr(line, "/libg.so")) continue;
    uint64_t start = strtoull(line, NULL, 16);
    if (!base || start < base) base = start;
  }
  fclose(maps);
  return base;
}

static uint64_t now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: comp_probe PID SAMPLES INTERVAL_MS ADDR [ADDR...]\n");
    return 2;
  }
  int pid = atoi(argv[1]), samples = atoi(argv[2]), interval = atoi(argv[3]);
  if (samples < 1 || samples > 10000 || interval < 10 || interval > 5000) return 2;
  char path[64];
  snprintf(path, sizeof(path), "/proc/%d/mem", pid);
  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 3;
  uint64_t base = libg_base(pid);
  if (!base) return 4;
  uint64_t start = now_ms();
  for (int s = 0; s < samples; ++s) {
    uint64_t began = now_ms();
    for (int a = 4; a < argc; ++a) {
      uint64_t obj = UNTAG(strtoull(argv[a], NULL, 0));
      uint8_t raw[0x168];
      if (!rd(obj, raw, sizeof(raw))) continue;
      int32_t x, y, state, deploy, count;
      uint64_t array;
      memcpy(&x, raw + 0x7c, 4);
      memcpy(&y, raw + 0x80, 4);
      memcpy(&state, raw + 0x11c, 4);
      memcpy(&deploy, raw + 0x15c, 4);
      memcpy(&array, raw + 0x18, 8);
      memcpy(&count, raw + 0x24, 4);
      printf("{\"t\":%" PRIu64 ",\"obj\":\"0x%" PRIx64 "\",\"x\":%d,\"y\":%d,\"state\":%d,"
             "\"deploy\":%d,\"count\":%d,\"comps\":[",
             now_ms() - start, obj, x, y, state, deploy, count);
      if (array && count > 0 && count <= MAX_COMPONENTS) {
        uint64_t comps[MAX_COMPONENTS];
        if (rd(array, comps, (size_t)count * 8)) {
          int first = 1;
          for (int i = 0; i < count; ++i) {
            uint64_t comp = UNTAG(comps[i]);
            uint8_t c[0x30];
            if (!comp || !rd(comp, c, sizeof(c))) continue;
            uint64_t vt, owner, p10;
            int32_t w20, w24, w28, w1e0 = -99999;
            memcpy(&vt, c + 0x00, 8);
            memcpy(&owner, c + 0x08, 8);
            memcpy(&p10, c + 0x10, 8);
            memcpy(&w20, c + 0x20, 4);
            memcpy(&w24, c + 0x24, 4);
            memcpy(&w28, c + 0x28, 4);
            // Only the movement component is that large; a failed read just means "not it".
            rd(comp + 0x1e0, &w1e0, 4);
            vt = UNTAG(vt);
            printf("%s{\"i\":%d,\"vt\":\"0x%" PRIx64 "\",\"owner\":%d,\"p10\":\"0x%" PRIx64
                   "\",\"w20\":%d,\"w24\":%d,\"w28\":%d,\"w1e0\":%d}",
                   first ? "" : ",", i, vt >= base ? vt - base : vt, UNTAG(owner) == obj,
                   (uint64_t)UNTAG(p10), w20, w24, w28, w1e0);
            first = 0;
          }
        }
      }
      printf("]}\n");
    }
    fflush(stdout);
    uint64_t spent = now_ms() - began;
    if (spent < (uint64_t)interval) usleep((useconds_t)(interval - spent) * 1000);
  }
  close(fd);
  return 0;
}
