#ifndef RANDOM_H
#define RANDOM_H

#include <time.h>
#include <stdlib.h>

static inline void init_random();
static inline void init_random_seed(unsigned int seed);
static inline double sample_uniform(double lower, double upper);
static inline int sample_uniform_int(int lower, int upper);

static inline double sample_uniform(double lower, double upper)
{
    return lower + ((double)random() / (double)RAND_MAX) * (upper - lower);
}

static inline void init_random()
{
    srandom(time(NULL));
}


static inline void init_random_seed(unsigned int seed)
{
    srandom(seed);
}

static inline int sample_uniform_int(int lower, int upper)
{
    return lower + (random() % (upper - lower + 1));
}

#endif