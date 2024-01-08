//
// Created by alexzms on 2024/1/2.
//

#ifndef RAYTRACING_CUDA_RAY_H
#define RAYTRACING_CUDA_RAY_H

#include "vec3.h"

namespace rt_cuda {
    class ray {
    public:
        __device__ ray() = default;
        __device__ ray(const vec3<float>& A, const vec3<float>& B): _origin(A), _direction(B) {}
        __device__ vec3<float> origin() const { return _origin; }
        __device__ vec3<float> direction() const {return _direction; }

    private:
        vec3<float> _origin, _direction;
    };
}

#endif //RAYTRACING_CUDA_RAY_H
