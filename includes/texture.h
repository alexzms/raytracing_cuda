//
// Created by alexzms on 2024/1/8.
//

#ifndef RAYTRACING_CUDA_TEXTURE_H
#define RAYTRACING_CUDA_TEXTURE_H

namespace rt_cuda {
    class texture_base {
    public:
        virtual __device__ ~texture_base() = default;
        [[nodiscard]] virtual __device__ color3f value(float u, float v, const point3f &p) const = 0;
    };

    class textures_wrapper {
    public:
        __host__ textures_wrapper() = default;
        __host__ explicit textures_wrapper(size_t capacity) {
            _h_d_t.reserve(capacity);
        }

        __host__ ~textures_wrapper() {
            printf("textures destroyed");
            for (auto & i : _h_d_t) {
                delete i;
            }
        }

        __host__ void store_inside(texture_base* d_content) {
            _h_d_t.emplace_back(d_content);
        }

        __host__ void store_inside(texture_base** h_d_contents, size_t len) {
            for (size_t i = 0; i < len; ++i) {
                store_inside(h_d_contents[i]);
            }
        }

        __host__ texture_base* get_d_texture(size_t index = 0) {
            if (index < _h_d_t.size()) {
                return _h_d_t[index];
            } else {
                return nullptr; // or handle the out-of-bounds case
            }
        }

    private:
        std::vector<texture_base*> _h_d_t;
    };

    class solid_color: public texture_base {
    public:
        __device__ explicit solid_color(const color3f& val): _color(val) {}
        __device__ solid_color(unsigned char r, unsigned char g, unsigned char b): _color(r, g, b) {}
        [[nodiscard]] __device__ color3f value(float u, float v, const point3f &p) const override {
            return _color;
        }

    private:
        color3f _color;
    };
}

#endif //RAYTRACING_CUDA_TEXTURE_H
