#ifndef POSTPROCESS_CLS_H
#define POSTPROCESS_CLS_H

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "common.hpp"

using namespace cls;

void cuda_postprocess(
    std::vector<Object>& objs, const float* d_output, int num_classes);

#endif // POSTPROCESS_CLS_H