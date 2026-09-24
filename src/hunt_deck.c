// Breadth-first hunt for an 8-card deck array anywhere near the battle graph.
//
// Reasoning: the simulation does not need the opponent's deck (each command carries its own
// card LogicData), so it may not be in the player object at all. But a replay/battle record
// must contain both decks to reproduce the match, and the x86 sandbox binding lists
// manager.replay_data at +0x78. So: start from every root we know, walk pointers to a bounded
// depth, and report any node holding several card-id values (25000000..29999999).
//
// Collect-then-scan: bounded node count, one 0x200 read per node, then stop.
//
// usage: hunt_deck PID MANAGER_RVA CTX_OFF [MAX_DEPTH] [MAX_NODES]
#define _GNU_SOURCE
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define UNTAG(p) ((p) & 0x00FFFFFFFFFFFFFFULL)
#define NODE 0x200
#define MAX_QUEUE 20000

static int fd;
static uint8_t buffer[NODE];

static int rd(uint64_t a, void *o, size_t n) {
  return pread(fd, o, n, (off_t)UNTAG(a)) == (ssize_t)n;
}
static uint64_t u64(uint64_t a) { uint64_t v = 0; rd(a, &v, 8); return v; }

static int plausible(uint64_t p) {
  uint64_t q = UNTAG(p);
  return q >= 0x10000 && q < 0x0000800000000000ULL && (q & 7) == 0;
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

typedef struct { uint64_t address; int depth; uint64_t parent; int parent_off; } Node;
static Node queue[MAX_QUEUE];
static uint64_t seen[MAX_QUEUE];
static int queue_len, seen_len;

static int already(uint64_t address) {
  for (int i = 0; i < seen_len; ++i) if (seen[i] == UNTAG(address)) return 1;
  return 0;
}

static void push(uint64_t address, int depth, uint64_t parent, int parent_off) {
  if (!plausible(address) || already(address) || queue_len >= MAX_QUEUE) return;
  seen[seen_len++] = UNTAG(address);
  queue[queue_len++] = (Node){address, depth, parent, parent_off};
}

int main(int argc, char **argv) {
  if (argc < 4 || argc > 6) {
    fprintf(stderr, "usage: hunt_deck PID MANAGER_RVA CTX_OFF [MAX_DEPTH] [MAX_NODES]\n");
    return 2;
  }
  int pid = atoi(argv[1]);
  uint64_t rva = strtoull(argv[2], NULL, 0), ctx_off = strtoull(argv[3], NULL, 0);
  int max_depth = argc > 4 ? atoi(argv[4]) : 3;
  int max_nodes = argc > 5 ? atoi(argv[5]) : 4000;
  if (max_nodes > MAX_QUEUE) max_nodes = MAX_QUEUE;

  uint64_t base = libg_base(pid);
  if (!base) return 3;
  char path[64];
  snprintf(path, sizeof(path), "/proc/%d/mem", pid);
  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 4;

  uint64_t root = u64(base + rva), context = u64(root + ctx_off);
  uint64_t battle = u64(context + 0x90), world = u64(battle + 0xA8);
  printf("{\"root\":\"0x%" PRIx64 "\",\"context\":\"0x%" PRIx64 "\",\"battle\":\"0x%" PRIx64
         "\",\"world\":\"0x%" PRIx64 "\",\"root_0x78\":\"0x%" PRIx64 "\",\"hits\":[",
         root, context, battle, world, u64(root + 0x78));

  push(root, 0, 0, -1);
  push(context, 0, 0, -1);
  push(battle, 0, 0, -1);
  push(world, 0, 0, -1);

  int first = 1, scanned = 0;
  for (int index = 0; index < queue_len && scanned < max_nodes; ++index) {
    Node node = queue[index];
    if (!rd(node.address, buffer, NODE)) continue;
    ++scanned;

    int cards = 0, best_run = 0, run = 0, run_start = -1, best_start = -1;
    for (int off = 0; off + 4 <= NODE; off += 4) {
      int32_t value;
      memcpy(&value, buffer + off, 4);
      int is_card = value >= 25000000 && value <= 29999999;
      cards += is_card;
      if (is_card) {
        if (!run) run_start = off;
        ++run;
        if (run > best_run) { best_run = run; best_start = run_start; }
      } else {
        run = 0;
      }
    }
    if (cards >= 4) {
      printf("%s{\"address\":\"0x%" PRIx64 "\",\"depth\":%d,\"parent\":\"0x%" PRIx64
             "\",\"parent_off\":\"0x%x\",\"card_values\":%d,\"longest_run\":%d,"
             "\"run_at\":\"0x%x\",\"run\":[",
             first ? "" : ",", node.address, node.depth, node.parent,
             node.parent_off < 0 ? 0 : node.parent_off, cards, best_run,
             best_start < 0 ? 0 : best_start);
      for (int k = 0; k < best_run && k < 16; ++k) {
        int32_t value;
        memcpy(&value, buffer + best_start + k * 4, 4);
        printf("%s%d", k ? "," : "", value);
      }
      printf("]}");
      first = 0;
    }
    if (node.depth < max_depth) {
      for (int off = 0; off + 8 <= NODE; off += 8) {
        uint64_t pointer;
        memcpy(&pointer, buffer + off, 8);
        push(pointer, node.depth + 1, node.address, off);
      }
    }
  }
  printf("],\"nodes_scanned\":%d,\"queued\":%d,\"max_depth\":%d}\n",
         scanned, queue_len, max_depth);
  close(fd);
  return 0;
}
