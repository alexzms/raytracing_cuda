//
// Created by alexzms on 2024/1/4.
//

#ifndef RAYTRACING_CUDA_CAMERA_H
#define RAYTRACING_CUDA_CAMERA_H

#include "vec3.h"
#include "hittable.h"
#include "hittable_list.h"
#include "fstream"
#include "material.h"

namespace rt_cuda {
    class camera {
    public:
        unsigned int samples_per_pixel = 9;                   // pixel sample time, large for better visual effects
        float vfov = 90;                                       // fov_height, width  will be calculated based on ratio
        float exposure_time = 1;                               // motion blur exposure time
        point3f lookfrom = point3f{0, 0, -1};      // look from position
        point3f lookat = point3f{0, 0, 0};         // look at position
        vec3f vup = vec3f{0, 1, 0};                // camera vup vector, actually it controls the rotation
        color3f background_color{0.2, 0.2, 0.2};  // {127, 179, 255}

        __device__ explicit camera(): image_height(0), viewport_width(0.0) {}
        __device__  ~camera() = default;

        __device__ void set_camera_parameter(int width, int height) {
            this->image_height = height;
            this->image_width = width;
        }

        __device__ void set_focus_parameter(float angle, float dist = 10.0) {
            this->defocus_angle = angle;
            this->focus_dist = dist;
            if (initialized) {
                refocus();
            }
        }


        __device__ void initialize() {
            if (initialized) return;
            initialized = true;

            reciprocal_sqrt_spp = 1.0f / sqrt(static_cast<float>(samples_per_pixel));

            camera_center = lookfrom;
            focal_length = focus_dist;

            w_ = normalize(lookfrom - lookat);                                   // opposite direction to view direction
            u = normalize(cross(vup, w_));
            v = cross(w_, u);

            auto theta = utilities::degree_to_radian(vfov);
            auto h = tan(theta / 2);                                     // h is height for unit focal_length
            viewport_height = 2 * h * focal_length;                                // * focal_length, it's easy to think
            viewport_width = viewport_height * (static_cast<float>(image_width) / image_height);

            viewport_u = viewport_width * u;
            viewport_v = viewport_height * -v;

            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            viewport_upper_left = camera_center - (focal_length * w_) - viewport_u / 2 - viewport_v / 2;
            pixel00_location = viewport_upper_left + 0.5 * pixel_delta_u + 0.5 * pixel_delta_v;

            auto defocus_radius = focus_dist * tan(utilities::degree_to_radian(defocus_angle / 2));
            defocus_disk_u = u * defocus_radius;                                    // camera defocus plane, basis vector
            defocus_disk_v = v * defocus_radius;
            printf("[rt_cuda][info] camera initialized\n");
            initialized = true;
        }

        __device__ static inline float linear_to_gamma(float val) {
            return std::sqrt(val);
        }

        __device__ ray get_ray_defocus_monte_carlo(curandState* state, unsigned int w,
                                                   unsigned int h, unsigned i, unsigned j) const
        {
            auto pixel_center = pixel00_location + h * pixel_delta_v + w * pixel_delta_u;
            auto pixel_random = pixel_center + pixel_sample_square_monte_carlo(state, i, j);

            auto ray_origin = (defocus_angle <= 0) ? camera_center : defocus_disk_sample(state);
            auto ray_direction = pixel_random - ray_origin;

            return ray{ray_origin, ray_direction};
        }

        [[nodiscard]] __device__ color3f ray_color(curandState* state, const ray &r,
                                                   unsigned int remain_depth, const hittable* world) const
        {
            if (remain_depth <= 0) return color3f{0, 0, 0};  // exceeds depths limit
            hit_record rec;
            if (!world->hit(r, interval(0.0001f, CONSTANT_INF), rec))
                return background_color;                                  // no hit -> return background color
//            else
//                return {255, 255, 255};
            color3f emission_color = rec.surface_material->emitted(state, rec.u, rec.v, rec.p);
//            return emission_color;
            ray scatter_ray;
            color3f attenuation;
            float pdf = 0.0;
            if (!rec.surface_material->scatter(state, r, rec, attenuation, scatter_ray, pdf))
                return emission_color;                                    // no scatter, just emission

            // probability of getting scatter_ray
//        float scattering_pdf = rec.surface_material->scattering_pdf(r, rec, scatter_ray);
//        float sample_pdf = scattering_pdf;                              // relative prob of sampling this ray
//
//        color scatter_color = attenuation
//                * scattering_pdf * ray_color(scatter_ray, remain_depth - 1, world) / sample_pdf;
            // naive way
            color3f scatter_color = attenuation * ray_color(state, scatter_ray, remain_depth - 1, world);
            return emission_color + scatter_color;
        }


    private:
        int image_width = 1600;
        int image_height = 900;
        float focal_length = 1.0f;
        float viewport_height = 2.0f;
        float viewport_width;                                               // we need not read this before initialize()
        point3f camera_center = point3f{0, 0, 0};

        float reciprocal_sqrt_spp = 0.0f;

        vec3f u{}, v{}, w_{};                                                // camera coordinate basis

        vec3f defocus_disk_u{};                                              // camera defocus plane
        vec3f defocus_disk_v{};

        float defocus_angle = 0;
        float focus_dist = 10;

        vec3f viewport_u{};
        vec3f viewport_v{};

        vec3f pixel_delta_u{};
        vec3f pixel_delta_v{};                                              // the base vector, expands a linear space

        vec3f viewport_upper_left{};
        vec3f pixel00_location{};

        bool initialized = false;

        __device__ void refocus() {
            if (!initialized) initialize();
            focal_length = focus_dist;
            auto theta = utilities::degree_to_radian(vfov);
            auto h = tan(theta / 2);
            viewport_height = 2 * h * focal_length;
            viewport_width = viewport_height * (static_cast<float>(image_width) / image_height);
            viewport_u = viewport_width * u;
            viewport_v = viewport_height * -v;

            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            viewport_upper_left = camera_center - (focal_length * w_) - viewport_u / 2 - viewport_v / 2;
            pixel00_location = viewport_upper_left + 0.5 * pixel_delta_u + 0.5 * pixel_delta_v;

            auto defocus_radius = focus_dist * tan(utilities::degree_to_radian(defocus_angle / 2));
            defocus_disk_u = u * defocus_radius;                                    // camera defocus plane, basis vector
            defocus_disk_v = v * defocus_radius;
        }

        __device__ point3f pixel_sample_square_monte_carlo(curandState* state, unsigned i, unsigned j) const {
            auto px = -0.5f + (i + utilities::random_float_d(state)) * reciprocal_sqrt_spp;
            auto py = -0.5f + (j + utilities::random_float_d(state)) * reciprocal_sqrt_spp;      // from -0.5~0.5

            return (px * pixel_delta_u) + (py * pixel_delta_v);                   // multiply the base vector
        }

        [[nodiscard]] __device__ vec3f defocus_disk_sample(curandState *state) const {
            auto p = random_in_unit_disk_d(state);
            return camera_center + p[0] * defocus_disk_u + p[1] * defocus_disk_v;
        }


        [[nodiscard]] __device__ point3f pixel_sample_square(curandState* state) const {
            auto px = utilities::random_float_d(state, -0.5, 0.5);
            auto py = utilities::random_float_d(state, -0.5, 0.5);      // from -0.5~0.5

            return (px * pixel_delta_u) + (py * pixel_delta_v);                   // multiply the base vector
        }

    };

    namespace kernel_funcs {
        __global__ void render
                (uchar3* d_ptr, const hittable* world, const camera* cam, unsigned full_w, unsigned full_h, unsigned sqrt_spp, unsigned max_depth) {
            unsigned width = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned height = threadIdx.y + blockIdx.y * blockDim.y;

            if (width >= full_w || height >= full_h) return;
            curandState state;
            curand_init(clock64(), width+height, 0, &state);

            unsigned index = width + height * full_w;
            vec3<float> pixel_color{0, 0, 0};
            for (unsigned i = 0; i != sqrt_spp; ++i) {
                for (unsigned j = 0; j != sqrt_spp; ++j) {
                    auto r = cam->get_ray_defocus_monte_carlo(&state, width, height, i, j);
                    pixel_color = pixel_color + cam->ray_color(&state, r, max_depth, world);
                }
            }
            // normalize (255 and spp)
            pixel_color /= static_cast<float>(sqrt_spp * sqrt_spp);
            // gamma correction
            pixel_color[0] = camera::linear_to_gamma(pixel_color[0]);
            pixel_color[1] = camera::linear_to_gamma(pixel_color[1]);
            pixel_color[2] = camera::linear_to_gamma(pixel_color[2]);

            uchar3 result = {static_cast<unsigned char>(pixel_color[0] * 255),
                               static_cast<unsigned char>(pixel_color[1] * 255),
                               static_cast<unsigned char>(pixel_color[2] * 255)};
            // write to d_ptr.
            d_ptr[index] = result;
        }
    }
    class camera_wrapper {
    public:
        camera_wrapper(camera* d_cam, GLsizei width, GLsizei height, uchar3* d_ptr, unsigned spp):
            _d_camera(d_cam), _spp(spp), _sqrt_spp(std::sqrt(spp)), _width(width), _height(height), _d_ptr(d_ptr)
            {}
        void render(hittable* d_world, dim3 block = dim3{10, 10}, unsigned max_depth = 50) {
            _block = block;
            _grid = dim3{(_width + _block.x - 1) / _block.x, (_height + _block.y - 1) / _block.y};
            printf("block: (%d, %d), grid: (%d, %d)\n", _block.x, _block.y, _grid.x, _grid.y);

            kernel_funcs::render<<<_grid, _block>>>(_d_ptr, d_world,
                                                    _d_camera, _width, _height,
                                                    _sqrt_spp, max_depth);
//            CHECK_ERROR(cudaDeviceSynchronize());
        }

//        friend __global__ void kernel_funcs::render (uchar3* d_ptr, const hittable* world,
//                                                     const camera* cam, unsigned full_w, unsigned full_h,
//                                                     unsigned sqrt_spp, unsigned max_depth);
    private:
        unsigned _spp, _sqrt_spp;
        uchar3* _d_ptr;
        dim3 _grid{}, _block{};
        GLsizei _width, _height;
        camera* _d_camera{};
    };
}

#endif //RAYTRACING_CUDA_CAMERA_H
