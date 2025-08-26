#ifndef TEST_CASE_H
#define TEST_CASE_H

#include "cgrad_test/config.h"
#include "cgrad_test/test_result.h"
#include "cgrad_test/test_case.h"

typedef void (*test_case)(struct test_result *const result);

#endif