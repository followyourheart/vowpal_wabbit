#ifndef MS_H
#define MS_H

#include "example.h"
#include "parser.h"

struct prediction {
  float p;
  size_t example_number;
};

struct global_prediction {
  float p;
  float weight;
};

const size_t multindex = 5;
extern size_t send_count;
extern size_t recv_count;

int really_read(int sock, void* in, size_t count);
int receive_features(parser* p, void* ex);
void send_prediction(int sock, prediction &pred);
bool blocking_get_prediction(int sock, prediction &p);
void send_global_prediction(int sock, global_prediction p);
bool blocking_get_global_prediction(int sock, global_prediction &p);

#endif