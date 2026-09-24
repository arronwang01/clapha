// Low-latency touch injection through the kernel input device.
//
// `input tap` spawns a fresh app_process (Java) per call, which is where the 150-400 ms goes.
// This opens /dev/input/eventN once, stays resident, and writes MT protocol B events directly,
// so a placement costs a couple of writes instead of two process launches.
//
// This is the ordinary Android input path, the same one `input tap` eventually reaches --
// it does NOT touch the game process. No injection, no hooks, no game memory written.
//
// Commands on stdin, one per line:
//   tap X Y                      single tap
//   place X0 Y0 X1 Y1 GAP_MS     card tap, gap, then placement tap
//   quit
// Prints one line per command with the elapsed microseconds, so latency is measurable.
//
// usage: fast_tap /dev/input/eventN
#define _GNU_SOURCE
#include <fcntl.h>
#include <linux/input.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int fd;

static uint64_t now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static void emit(uint16_t type, uint16_t code, int32_t value) {
  struct input_event ev;
  memset(&ev, 0, sizeof(ev));
  ev.type = type;
  ev.code = code;
  ev.value = value;
  if (write(fd, &ev, sizeof(ev)) != (ssize_t)sizeof(ev))
    perror("write");
}

static void sync_report(void) { emit(EV_SYN, SYN_REPORT, 0); }

static int tracking = 1;

static void touch_down(int x, int y) {
  emit(EV_ABS, ABS_MT_SLOT, 0);
  emit(EV_ABS, ABS_MT_TRACKING_ID, tracking++);
  emit(EV_ABS, ABS_MT_POSITION_X, x);
  emit(EV_ABS, ABS_MT_POSITION_Y, y);
  emit(EV_KEY, BTN_TOUCH, 1);
  sync_report();
}

static void touch_up(void) {
  emit(EV_ABS, ABS_MT_SLOT, 0);
  emit(EV_ABS, ABS_MT_TRACKING_ID, -1);
  emit(EV_KEY, BTN_TOUCH, 0);
  sync_report();
}

/* A touch that goes down and up inside one frame can be dropped, so hold briefly. */
static void tap(int x, int y, int hold_ms) {
  touch_down(x, y);
  usleep((useconds_t)hold_ms * 1000);
  touch_up();
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "/dev/input/event1";
  fd = open(path, O_WRONLY | O_CLOEXEC);
  if (fd < 0) { perror(path); return 2; }
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("{\"ready\":\"%s\"}\n", path);

  char line[256];
  while (fgets(line, sizeof(line), stdin)) {
    uint64_t started = now_us();
    int x0, y0, x1, y1, gap;
    if (sscanf(line, "place %d %d %d %d %d", &x0, &y0, &x1, &y1, &gap) == 5) {
      tap(x0, y0, 34);
      usleep((useconds_t)gap * 1000);
      tap(x1, y1, 34);
      printf("{\"place\":[%d,%d,%d,%d],\"us\":%llu}\n", x0, y0, x1, y1,
             (unsigned long long)(now_us() - started));
    } else if (sscanf(line, "tap %d %d", &x0, &y0) == 2) {
      tap(x0, y0, 34);
      printf("{\"tap\":[%d,%d],\"us\":%llu}\n", x0, y0,
             (unsigned long long)(now_us() - started));
    } else if (!strncmp(line, "quit", 4)) {
      break;
    } else {
      printf("{\"error\":\"bad command\"}\n");
    }
  }
  close(fd);
  return 0;
}
