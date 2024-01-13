//
// Created by alexzms on 2024/1/8.
//

#ifndef RAYTRACING_CUDA_AABB_H
#define RAYTRACING_CUDA_AABB_H

#include "interval.h"
#include "vec3.h"
#include "ray.h"

namespace rt_cuda {
    class aabb {
    public:
        interval x, y, z;
        __device__ aabb() = default;                                      // default: intervals are all universe
        __device__ aabb(const interval& ix, const interval& iy, const interval& iz): x(ix), y(iy), z(iz) {}
        __device__ aabb(const point3f& p1, const point3f& p2) {
            x = interval(fmin(p1[0], p2[0]), fmax(p1[0], p2[0]));
            y = interval(fmin(p1[1], p2[1]), fmax(p1[1], p2[1]));
            z = interval(fmin(p1[2], p2[2]), fmax(p1[2], p2[2]));
        }
        __device__ aabb(const aabb& b1, const aabb& b2) {
            x = interval{b1.x, b2.x};                                       // interval constructor will automatically
            y = interval{b1.y, b2.y};                                       // merge the two intervals
            z = interval{b1.z, b2.z};
        }


        [[nodiscard]] __device__  const interval& axis(int dim) const {
            if (dim == 0) return x;
            if (dim == 1) return y;
            if (dim == 2) return z;
            return z;
        }

        __device__ void this_pad(double delta = 0.0001) {
            this->x = (x.size() >= delta) ? x.expand(delta): x;
            this->y = (y.size() >= delta) ? y.expand(delta): y;
            this->z = (z.size() >= delta) ? z.expand(delta): z;
        };

        [[nodiscard]] __device__ aabb pad(double delta = 0.0001) const {
            interval new_x = (x.size() >= delta) ? x.expand(delta): x;
            interval new_y = (y.size() >= delta) ? y.expand(delta): y;
            interval new_z = (z.size() >= delta) ? z.expand(delta): z;
            return {new_x, new_y, new_z};
        }

        // This version works better with compiler, although it's mathematically the same with hit_my_version
        [[nodiscard]] __device__ bool hit(const ray& r, interval ray_t) const {
            for (int a = 0; a < 3; a++) {
                auto invD = 1 / r.direction()[a];                             // store the 1/direction
                auto orig = r.origin()[a];                                    // also store this
                auto t0 = (axis(a).min - orig) * invD;                   // don't need to calculate those again
                auto t1 = (axis(a).max - orig) * invD;
                if (invD < 0)                                                         // the case of negative ray
                    swap(t0, t1);                                          // code like this avoids min/max()
                if (t0 > ray_t.min) ray_t.min = t0;                                   // use 'if' instead of fmax, fmin
                if (t1 < ray_t.max) ray_t.max = t1;                                   // is more compiler friendly I guess
                if (ray_t.max <= ray_t.min)                                           // corner is considered no
                    return false;
            }
            return true;
        }
    private:
        __device__ void swap(float& a, float& b) const {
            auto temp = a;
            a = b;
            b = temp;
        }
        // This version is mathematically the same with hit(), but it's not as compiler friendly as hit()
        [[deprecated]][[nodiscard]] __device__  bool hit_my_version(const ray& r, interval ray_t) const {
            for (int axis_i = 0; axis_i != 3; ++axis_i) {
                auto t0 = fmin( (axis(axis_i).min - r.origin().x())/r.direction().x() ,// t0 = min( (x0 - Ax)/bx),
                                (axis(axis_i).max - r.origin().x())/r.direction().x());      //           (x1 - Ax)/bx)
                auto t1 = fmax( (axis(axis_i).min - r.origin().x())/r.direction().x() ,// t1 = min( (x0 - Ax)/bx),
                                (axis(axis_i).max - r.origin().x())/r.direction().x());      //           (x1 - Ax)/bx)
                ray_t.min = fmax(t0, ray_t.min);
                ray_t.max = fmin(t1, ray_t.max);
                if (ray_t.max <= ray_t.min) return false;                                           // corner is considered no
            }
            return true;
        }

    };

    __device__ aabb operator+(const aabb& bbox, const vec3f& offset) {
        return {bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z()};
    }

    __device__ aabb operator+(const vec3f& offset, const aabb& bbox) {
        return bbox + offset;
    }
}

#endif //RAYTRACING_CUDA_AABB_H
