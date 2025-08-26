#ifndef CGRAD_ASSERT_H
#define CGRAD_ASSERT_H

#include "cgrad_test/test_result.h"

static inline void test_result_set_error(struct test_result *const result, cgrad_test_result_error err, const char *msg);

#define ASSERT_TRUE(expr, msg)                               \
    do                                                       \
    {                                                        \
        if (!expr)                                           \
        {                                                    \
            test_result_set_error(result, TEST_FAILED, msg); \
            return;                                          \
        }                                                    \
    } while (0)

static inline void test_result_set_error(struct test_result *const result, cgrad_test_result_error err, const char *msg)
{
    if (!result)
    {
        return;
    }

    result->err = err;

    const char NULL_TERMINATOR = 0;
    memccpy(result->msg, msg, NULL_TERMINATOR, TEST_RESULT_MSG_MAX_SIZE);
}

#endif