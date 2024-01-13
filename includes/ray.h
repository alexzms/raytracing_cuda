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
        __device__ ray(const vec3<float>& origin, const vec3<float>& direction): _origin(origin), _direction(direction) {}
        __device__ vec3<float> origin() const { return _origin; }
        __device__ vec3<float> direction() const {return _direction; }

        __device__ vec3<float> at(float t) const { return _origin + t * _direction; }

    private:
        vec3<float> _origin, _direction;
    };
}

#endif //RAYTRACING_CUDA_RAY_H
