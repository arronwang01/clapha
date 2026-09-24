// Why does the opponent's 8-card deck not resolve?
//
// Upstream read_visible_deck walks, per player:
//   player+0x10 -> context ; context+0x98 -> root ; player+0x78 -> side
//   root+0x30 + side*8 -> avatar ; avatar+0x00/+0x04 -> account hi/lo
//   root+0x60 -> avatar count (1..6)
//   root+0x88 + identity*8 -> owner ; owner+0x20 -> entries ; owner+0x2c -> entry count (==8)
//   entries[i] -> entry ; entry+0x10 -> data ; data+0x40 -> card id ; entry+0x1c -> form
// It returns 0 on the first mismatch, so a single changed offset looks like "hidden".
// This dumps every step for BOTH players plus raw windows, to see where it stops.
//
// usage: deck_probe PID MANAGER_RVA CTX_OFF
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

static void dump_words(const char *label, uint64_t base, int from, int to) {
  printf("    \"%s\":[", label);
  for (int off = from; off < to; off += 4) {
    if (off != from) putchar(',');
    printf("[\"0x%x\",%d]", off, i32(base + off));
  }
  printf("],\n");
}

int main(int argc, char **argv) {
  if (argc != 4) { fprintf(stderr, "usage: deck_probe PID MANAGER_RVA CTX_OFF\n"); return 2; }
  int pid = atoi(argv[1]);
  uint64_t rva = strtoull(argv[2], NULL, 0), ctx_off = strtoull(argv[3], NULL, 0);
  uint64_t base = libg_base(pid);
  if (!base) return 3;
  char path[64];
  snprintf(path, sizeof(path), "/proc/%d/mem", pid);
  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 4;

  uint64_t root = u64(base + rva);
  uint64_t context = u64(root + ctx_off);
  uint64_t battle = u64(context + 0x90);
  uint64_t player_state = u64(battle + 0xA8);
  printf("{\n  \"battle\":\"0x%" PRIx64 "\",\"player_state\":\"0x%" PRIx64 "\",\n  \"players\":[\n",
         battle, player_state);

  for (int side_index = 0; side_index < 2; ++side_index) {
    uint64_t player = u64(player_state + 0xE0 + (uint64_t)side_index * 8);
    uint64_t pcontext = u64(player + 0x10);
    uint64_t proot = u64(pcontext + 0x98);
    int32_t side = i32(player + 0x78);
    int32_t avatar_count = i32(proot + 0x60);
    printf("    {\"index\":%d,\"player\":\"0x%" PRIx64 "\",\"context\":\"0x%" PRIx64
           "\",\"root\":\"0x%" PRIx64 "\",\"side_field\":%d,\"avatar_count_0x60\":%d,\n",
           side_index, player, pcontext, proot, side, avatar_count);

    uint64_t own_avatar = (side >= 0 && side < 6) ? u64(proot + 0x30 + (uint64_t)side * 8) : 0;
    int32_t own_hi = own_avatar ? i32(own_avatar) : 0, own_lo = own_avatar ? i32(own_avatar + 4) : 0;
    printf("     \"own_avatar\":\"0x%" PRIx64 "\",\"own_account\":[%d,%d],\n",
           own_avatar, own_hi, own_lo);

    printf("     \"avatar_slots\":[");
    for (int i = 0; i < 6; ++i) {
      uint64_t a = u64(proot + 0x30 + (uint64_t)i * 8);
      if (i) putchar(',');
      printf("{\"i\":%d,\"addr\":\"0x%" PRIx64 "\",\"hi\":%d,\"lo\":%d}",
             i, a, a ? i32(a) : 0, a ? i32(a + 4) : 0);
    }
    printf("],\n");

    printf("     \"owner_slots\":[");
    for (int i = 0; i < 6; ++i) {
      uint64_t owner = u64(proot + 0x88 + (uint64_t)i * 8);
      uint64_t entries = owner ? u64(owner + 0x20) : 0;
      int32_t count = owner ? i32(owner + 0x2c) : -1;
      if (i) putchar(',');
      printf("{\"i\":%d,\"owner\":\"0x%" PRIx64 "\",\"entries\":\"0x%" PRIx64 "\",\"count_0x2c\":%d",
             i, owner, entries, count);
      if (entries && count > 0 && count <= 16) {
        printf(",\"cards\":[");
        for (int k = 0; k < count; ++k) {
          uint64_t entry = u64(entries + (uint64_t)k * 8);
          uint64_t data = entry ? u64(entry + 0x10) : 0;
          if (k) putchar(',');
          printf("%d", data ? i32(data + 0x40) : -1);
        }
        printf("]");
      }
      putchar('}');
    }
    printf("],\n");
    /* raw entry/data for the first two owner slots: is a card id hiding elsewhere? */
    for (int oi = 0; oi < 2; ++oi) {
      uint64_t owner = u64(proot + 0x88 + (uint64_t)oi * 8);
      uint64_t entries = owner ? u64(owner + 0x20) : 0;
      if (!entries) continue;
      uint64_t entry = u64(entries), data = entry ? u64(entry + 0x10) : 0;
      printf("     \"owner%d_entry0\":\"0x%" PRIx64 "\",\"owner%d_data0\":\"0x%" PRIx64 "\",\n",
             oi, entry, oi, data);
      char label[64];
      snprintf(label, sizeof(label), "owner%d_entry_words", oi);
      if (entry) dump_words(label, entry, 0x00, 0x60);
      snprintf(label, sizeof(label), "owner%d_data_words", oi);
      if (data) dump_words(label, data, 0x00, 0x80);
    }
    dump_words("player_words_0x00_0x40", player, 0x00, 0x40);
    dump_words("root_words_0x50_0xd0", proot, 0x50, 0xd0);
    printf("     \"end\":true}%s\n", side_index == 0 ? "," : "");
  }
  printf("  ]\n}\n");
  close(fd);
  return 0;
}
