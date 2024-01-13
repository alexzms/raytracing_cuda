//
// Created by alexzms on 2024/1/4.
//

#ifndef RAYTRACING_CUDA_HITTABLE_H
#define RAYTRACING_CUDA_HITTABLE_H

#include "interval.h"
#include "aabb.h"


namespace rt_cuda {
    class material_base;

    class hit_record {
    public:
        point3f p{};
        vec3f normal{};
        float t{};
        float u{}, v{};
        material_base* surface_material{};
        bool front_face{};

        __device__ hit_record() = default;

        __device__ inline void set_face_normal(const ray& r, const vec3f& outward_normal) {
            // outward_normal should have length 1
            front_face = dot(r.direction(), outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
        }
    };

    class hittable {
    public:
        __device__ virtual ~hittable() = default;
        __device__ virtual bool hit(const ray& r, const interval &inter, hit_record &rec) const = 0;
        [[nodiscard]] __device__ virtual aabb bounding_box() const = 0;
    };
}

#endif //RAYTRACING_CUDA_HITTABLE_H
