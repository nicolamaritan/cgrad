#ifndef TEST_RESULT_H
#define TEST_RESULT_H

#include "cgrad_test/config.h"
#include "cgrad_test/test_result.h"
#include "cgrad_test/test_result_error.h"
#include <string.h>

struct test_result
{
    cgrad_test_result_error err;
    char msg[TEST_RESULT_MSG_MAX_SIZE];
};

static inline void test_result_set_error(struct test_result *const result, cgrad_test_result_error err, const char *msg);

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