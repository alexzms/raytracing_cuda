//
// Created by alexzms on 2024/1/2.
//

#ifndef RAYTRACING_CUDA_VEC3_H
#define RAYTRACING_CUDA_VEC3_H

#include "cuda_helpers.h"

template <typename T>
class vec3 {
public:
    CUDA_CALLABLE_MEMBER vec3() : _e{0, 0, 0} {}

    CUDA_CALLABLE_MEMBER vec3(T e0, T e1, T e2) : _e{e0, e1, e2} {}
    CUDA_CALLABLE_MEMBER inline T x() const { return _e[0]; }
    CUDA_CALLABLE_MEMBER inline T y() const { return _e[1]; }
    CUDA_CALLABLE_MEMBER inline T z() const { return _e[2]; }
    CUDA_CALLABLE_MEMBER inline T& x() { return _e[2]; }
    CUDA_CALLABLE_MEMBER inline T& y() { return _e[2]; }
    CUDA_CALLABLE_MEMBER inline T& z() { return _e[2]; }



private:
    T _e[3];
};

#endif //RAYTRACING_CUDA_VEC3_H
