#ifndef RANDOM_H
#define RANDOM_H

#include <time.h>
#include <stdlib.h>

/**
 * @brief Initialize the random number generator with a time-based seed.
 *
 * Uses the current system time as seed, ensuring different
 * random sequences across program runs.
 */
static inline void init_random();

/**
 * @brief Initialize the random number generator with a fixed seed.
 *
 * Useful for reproducibility: the same seed will always produce
 * the same sequence of random numbers.
 *
 * @param seed Seed value for the random number generator.
 */
static inline void init_random_seed(unsigned int seed);

/**
 * @brief Sample a random floating-point number from a uniform distribution.
 *
 * Generates a double precision value uniformly distributed
 * in the interval [lower, upper].
 *
 * @param lower Lower bound of the interval (inclusive).
 * @param upper Upper bound of the interval (inclusive).
 * @return Random value in [lower, upper].
 */
static inline double sample_uniform(double lower, double upper);

/**
 * @brief Sample a random integer from a uniform distribution.
 *
 * Generates an integer uniformly distributed in the interval [lower, upper].
 *
 * @param lower Lower bound of the interval (inclusive).
 * @param upper Upper bound of the interval (inclusive).
 * @return Random integer in [lower, upper].
 *
 * @note Uses modulo arithmetic, which may introduce slight bias
 *       if (upper - lower + 1) does not divide evenly into RAND_MAX.
 */
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