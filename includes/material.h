//
// Created by alexzms on 2024/1/8.
//

#ifndef RAYTRACING_CUDA_MATERIAL_H
#define RAYTRACING_CUDA_MATERIAL_H

#include "memory"
#include "hittable.h"
#include "ray.h"
#include "texture.h"
#include "onb.h"

namespace rt_cuda {
    class material_base {
    public:
        virtual __device__ ~material_base() = default;
        virtual __device__ bool scatter
            (curandState* state, const ray& in, const hit_record& rec, color3f& color, ray& out, float& pdf) const = 0;

        virtual __device__ color3f emitted
            (curandState* state, float u, float v, const point3f& p) const
        {
            return {0, 0, 0};                                           // default: no emission
        }

        virtual __device__ float scatter_pdf
            (curandState* state, const ray& in, const hit_record& rec, const ray& scattered) const
        {
            return 0.0f;                                                            // default: not using scatter_pdf
        }
    };

    class materials_wrapper {
    public:
        __host__ materials_wrapper() = default;
        __host__ explicit materials_wrapper(size_t capacity) {
            _h_d_m.reserve(capacity);
        }

        __host__ ~materials_wrapper() {
            for (auto & i : _h_d_m) {
                delete i;
            }
        }

        __host__ void store_inside(material_base* d_content) {
            _h_d_m.push_back(d_content);
        }

        __host__ void store_inside(material_base** h_d_contents, size_t len) {
            for (size_t i = 0; i < len; ++i) {
                store_inside(h_d_contents[i]);
            }
        }

        __host__ material_base* get_d_texture(size_t index = 0) {
            if (index < _h_d_m.size()) {
                return _h_d_m[index];
            } else {
                return nullptr; // or handle the out-of-bounds case
            }
        }

    private:
        std::vector<material_base*> _h_d_m;
    };

    class lambertian : public material_base {
    public:
        __device__ explicit lambertian(texture_base* d_tex): _d_tex(d_tex) {}

        __device__ bool scatter(curandState* state, const ray& in,
                              const hit_record& rec, color3f& color,
                              ray& out, float& pdf) const override
        {
//                                                                            // already satisfying the scattering_pdf
            float a = 2 * CONSTANT_PI * curand_uniform(state);
            float z = -1 + 2 * curand_uniform(state);
            float r = sqrtf(1 - z * z);
//            vec3f scatter_direction = rec.normal + vec3f{r * cosf(a), r * sinf(a), z};
//            onb uvw;
//            uvw.build_from_normal(rec.normal);                                // lambertian distribution(cosine distri)
//            auto scatter_direction = uvw.local_to_global(random_cosine_direction_d(state));

//            if (true || scatter_direction.near_zero()) {
//                scatter_direction = rec.normal;
//            }
            out = ray{rec.p, rec.normal + vec3f{r * cosf(a), r * sinf(a), z}};
            color = _d_tex->value(rec.u, rec.v, rec.p);
//            pdf = dot(uvw.w(), out.direction()) / CONSTANT_PI;    // pdf=cos(theta)/pi
            return true;
        }

        __device__ float scatter_pdf(curandState* state, const ray& in,
                                   const hit_record& rec, const ray& scattered) const override
        {
            onb uvw;
            uvw.build_from_normal(rec.normal);
            return dot(uvw.w(), scattered.direction()) / CONSTANT_PI;
            auto normal = rec.normal;
            auto cos_theta = dot(normal, scattered.direction());
            return cos_theta > 0 ? cos_theta/CONSTANT_PI :  0;                           // pdf=cos(theta)/pi
        }

    private:
        texture_base* _d_tex;
    };
}

#endif //RAYTRACING_CUDA_MATERIAL_H
