//
// Created by alexzms on 2024/1/7.
//

#ifndef RAYTRACING_CUDA_UTILITIES_H
#define RAYTRACING_CUDA_UTILITIES_H

#include "random"
#include "curand.h"
#include "curand_kernel.h"

#define CONSTANT_PI (3.1415926535897932385f)
#define CONSTANT_INF (1e8f)
#define CONSTANT_EPSILON (1e-6f)

namespace rt_cuda::utilities {

    __device__ inline float degree_to_radian(float degree) {
        return degree * CONSTANT_PI / 180.0f;
    }

    __device__ float random_float_d(curandState* state) {
        return curand_uniform(state);
    }
    __device__ float random_float_d(curandState* state, float min, float max) {
        if (min == max) return min;
        return min + (max - min) * random_float_d(state);
    }
    __device__ float _random_float_d(curandState* local_rand_state) {
        return curand_uniform(local_rand_state);
    }}

#endif //RAYTRACING_CUDA_UTILITIES_H
