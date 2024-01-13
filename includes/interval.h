//
// Created by alexzms on 2024/1/8.
//

#ifndef RAYTRACING_CUDA_INTERVAL_H
#define RAYTRACING_CUDA_INTERVAL_H

#include "utilities.h"

namespace rt_cuda {
    class interval {
    public:
        float min, max;
        __device__ interval(): min(+CONSTANT_INF), max(-CONSTANT_INF) {}         // default interval is empty
        __device__ interval(float min_, float max_): min(min_), max(max_) {}                 // constructor below merges two interval
        __device__ interval(const interval& i1, const interval& i2):
                                min(fmin(i1.min, i2.min)), max(fmax(i1.max, i2.max)) {}

        [[nodiscard]] __device__ inline bool contains(float val) const {
            return min <= val && val <= max;
        }
        [[nodiscard]] __device__ inline bool surrounds(float val) const {
            return min < val && val < max;
        }
        [[nodiscard]] __device__ inline float clamp(float val) const {
            if (val < min) return min;
            if (val > max) return max;
            return val;
        }
        [[nodiscard]] __device__ inline float size() const {
            return max - min;
        }
        [[nodiscard]] __device__ inline interval expand(float delta) const {
            auto padding = delta / 2;
            return {min - padding, max + padding};
        }
        static const interval empty;
        static const interval universe;
    };

    __device__ interval operator+(const interval& ival, float displacement) {
        return {ival.min + displacement, ival.max + displacement};
    }

    __device__ interval operator+(float displacement, const interval& ival) {
        return ival + displacement;
    }
}

#endif //RAYTRACING_CUDA_INTERVAL_H
