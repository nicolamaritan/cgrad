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

#endif