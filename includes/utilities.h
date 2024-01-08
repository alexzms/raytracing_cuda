//
// Created by alexzms on 2024/1/7.
//

#ifndef RAYTRACING_CUDA_UTILITIES_H
#define RAYTRACING_CUDA_UTILITIES_H

#include "random"
#include "curand.h"
#include "curand_kernel.h"

namespace rt_cuda::utilities {
    constexpr static float M_PI = 3.1415926535897932385f;
    constexpr static float infinity = std::numeric_limits<float>::infinity();
    constexpr static float epsilon = 1e-8f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    __host__ __device__ inline float degree_to_radian(float degree) {
        return degree * M_PI / 180.0f;
    }
    __host__ float random_float_h() {
        return dis(gen);
    }
    __host__ float random_float_h(float min, float max) {
        if (min == max) return min;
        return min + (max - min) * random_float_h();
    }

    __device__ float random_float_d() {
                                                        // the performance of this function is not optimal(repeat init)
        static curandState local_rand_state;
        curand_init(clock64(), 0, 0, &local_rand_state);
        return curand_uniform(&local_rand_state);
    }
    __device__ float _random_float_d(curandState* local_rand_state) {
        return curand_uniform(local_rand_state);
    }}

#endif //RAYTRACING_CUDA_UTILITIES_H
