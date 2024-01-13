//
// Created by alexzms on 2024/1/11.
//

#ifndef RAYTRACING_CUDA_SPHERE_H
#define RAYTRACING_CUDA_SPHERE_H

#include "hittable.h"
#include "utility"
#include "vec3.h"
#include "material.h"

namespace rt_cuda {
    class sphere : public hittable {
    public:
        __device__ sphere(point3f center, float radius, material_base* d_material):
                _center(center), radius(radius), _d_material(d_material){
            auto half_edge = vec3f{radius, radius, radius};
            bbox = aabb{_center - half_edge, _center + half_edge};
        }

        __device__ bool hit(const ray& r, const interval& inter, hit_record &rec) const override {
            // a = dir . dir = ||dir||_2^2 = 1, h_b = dir . (origin-_center), c = ||origin- _center||_2^2 - radius^2
            // h_discriminant = h_b - a*c
            vec3 oc = r.origin() - _center;
            auto a = r.direction().length_sq();         // must be +
            auto half_b = dot(r.direction(), oc);  // if collides, must be -
            auto c = oc.length_sq() - radius * radius;  // not known, but normally it should be +
            auto discriminant = half_b * half_b - a * c;
            if (discriminant < 0) return false;
            auto sqrt_d = std::sqrt(discriminant);
            auto root = (-half_b - sqrt_d) / a;            // first we use the smaller solution(closest hit)
            if (!inter.surrounds(root)) {
                root = (-half_b + sqrt_d) / a;                     // switch to the larger solution
                if (!inter.surrounds(root)) {
                    return false;                                  // if it still not work, return false
                }
            }                                                      // now root must be within the range of t_min and t_max
            rec.t = root;
            rec.p = r.at(root);                                 // update the hit record
            rec.surface_material = _d_material;
            vec3 outward_normal = (rec.p - _center) / radius;
            rec.set_face_normal(r, outward_normal);
            get_sphere_uv(outward_normal, rec.u, rec.v);

            return true;
        }

        [[nodiscard]] __device__ aabb bounding_box() const override { return bbox; }

    private:
        point3f _center;
        float radius;
        aabb bbox;
        material_base* _d_material;

        __device__ static void get_sphere_uv(const point3f& p, float& u, float& v) {
            // p: a given point on the sphere of radius one, centered at the origin.
            // u: returned value [0,1] of angle around the Y _axis from X=-1.
            // v: returned value [0,1] of angle from Y=-1 to Y=+1.
            //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
            //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
            //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

            auto theta = acos(-p.y());
            auto phi = atan2(-p.z(), p.x()) + CONSTANT_PI;
            u = phi / (2 * CONSTANT_PI);
            v = theta / CONSTANT_PI;
        }
    };
}

#endif //RAYTRACING_CUDA_SPHERE_H
