//
// Created by alexzms on 2024/1/8.
//

#ifndef RAYTRACING_CUDA_HITTABLE_LIST_H
#define RAYTRACING_CUDA_HITTABLE_LIST_H

#include "vec3.h"

namespace rt_cuda {
    class hittable_list: public hittable {
    public:
        __device__ hittable_list(hittable** d_hittables, size_t length):
                            _d_hittables(d_hittables), _d_hittables_length(length) {}

        __device__ bool hit(const ray& r, const interval &inter, hit_record &rec) const override {
            hit_record temp_rec;
            interval temp_interval(inter);
            bool hit_any = false;

            for (int i = 0; i != _d_hittables_length; ++i) {
                if (_d_hittables[i]->hit(r, temp_interval, temp_rec)) {  // if the object can be hit within the
                    hit_any = true;                                               // range of t_min and closest_to_far
                    temp_interval.max = temp_rec.t;                               // closest_to_far is temp_interval.max
                    rec = temp_rec;
                }
            }

            return hit_any;
        }

        [[nodiscard]] __device__ aabb bounding_box() const override {
            return _bbox;
        }

    private:
        hittable** _d_hittables;
        size_t _d_hittables_length;
        aabb _bbox{};
    };
}

#endif //RAYTRACING_CUDA_HITTABLE_LIST_H
