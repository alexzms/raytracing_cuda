//
// Created by alexzms on 2024/1/8.
//

#ifndef RAYTRACING_CUDA_ONB_H
#define RAYTRACING_CUDA_ONB_H

#include "vec3.h"

namespace rt_cuda {
    class onb {
    public:
        __device__ onb() = default;                                    // constructor constructed all zero _axis, which is invalid

        __device__ vec3f &operator[](int index) { return _axis[index]; }

        __device__ vec3f operator[](int index) const { return _axis[index]; }

        [[nodiscard]] __device__ vec3f u() const { return _axis[0]; }

        [[nodiscard]] __device__ vec3f v() const { return _axis[1]; }

        [[nodiscard]] __device__ vec3f w() const { return _axis[2]; }

        [[nodiscard]] __device__ vec3f &u() { return _axis[0]; }

        [[nodiscard]] __device__ vec3f &v() { return _axis[1]; }

        [[nodiscard]] __device__ vec3f &w() { return _axis[2]; }

        __device__ vec3f local_to_global(float x, float y, float z) {
            return _axis[0] * x + _axis[1] * y + _axis[2] * z;
        }

        __device__ vec3f local_to_global(const vec3f &local_coords) {
            return _axis[0] * local_coords.x() + _axis[1] * local_coords.y() + _axis[2] * local_coords.z();
        }

        __device__ void build_from_normal(const vec3f &normal) {
            vec3f normalized_normal = normalize(normal);
            vec3f a = (fabs(normalized_normal.x()) > 0.9f) ? vec3f(0, 1, 0) : vec3f(1, 0, 0);
            vec3f v = normalize(cross(a, normalized_normal));
            vec3f u = cross(normalized_normal, v);
            _axis[0] = u;
            _axis[1] = v;
            _axis[2] = normalized_normal;
        }


    private:
        vec3f _axis[3];
    };
}

#endif //RAYTRACING_CUDA_ONB_H
