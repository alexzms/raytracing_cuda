//
// Created by alexzms on 2024/1/1.
//

#ifndef RAYTRACING_CUDA_UNSTRUCTURED_CONTENT_H
#define RAYTRACING_CUDA_UNSTRUCTURED_CONTENT_H

#ifndef VISUALIME_USE_CUDA
#define VISUALIME_USE_CUDA
#endif
#include "visualime/visualime.h"
#include "cuda_helpers.h"
#include "iostream"
#include "vector"
#include "algorithm"

__global__ static void render(uchar3* ptr, size_t size, size_t row_size, size_t column_size, unsigned char z) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= row_size || y >= column_size) return;

    size_t offset = x + y * blockDim.x * gridDim.x;

    ptr[offset].x = static_cast<unsigned char>(255 * x / row_size);
    ptr[offset].y = static_cast<unsigned char>(255 * y / column_size);
    ptr[offset].z = static_cast<unsigned char>(z);
}

class output_an_image {
public:
    output_an_image(GLsizei width, GLsizei height, uchar3* d_ptr, size_t ptr_size):
                                _width(width), _height(height), _d_ptr(d_ptr), _ptr_size(ptr_size) {
        std::cout << "[output_an_image][info] width: " << _width << ", height: " << _height << std::endl;
    }

    void run(unsigned char z = 51) const {
        cudaEvent_t start, stop;
        CHECK_ERROR(cudaEventCreate(&start));
        CHECK_ERROR(cudaEventCreate(&stop));
        CHECK_ERROR(cudaEventRecord(start, nullptr));
        size_t thread_x = 8, thread_y = 8;
        if (_width % thread_x != 0 || _height % thread_y != 0)
            std::cout << "[output_an_image][warn] _row_size % thread_x != 0 || _column_size % thread_y != 0" << std::endl;
        size_t block_x = _width / thread_x, block_y = _height / thread_y;
        dim3 grid(block_x, block_y);
        dim3 block(thread_x, thread_y);
        render<<<grid, block>>>(_d_ptr, _ptr_size, _width, _height, z);
        CHECK_ERROR(cudaDeviceSynchronize());
        CHECK_ERROR(cudaEventRecord(stop, nullptr));
        CHECK_ERROR(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CHECK_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        std::cout << "[output_an_image][info] render time: " << milliseconds << "ms" << std::endl;
    }

private:
    GLsizei _width;
    GLsizei _height;
    uchar3* _d_ptr;
    size_t _ptr_size;

    template<typename T>
    static void decompose_number(T k, T* a, T* b) {
        auto sqrt_k = static_cast<unsigned int>(std::ceil(std::sqrt(k)));
        while (sqrt_k != 0) {                                       // we don't care too much performance issue here
            if (k % sqrt_k == 0)                                    // because this function will only be called once
                { *a = sqrt_k; *b = k / sqrt_k; return; }
            sqrt_k -= 1;                                            // this loop "will" end, since when sqrt_k = 1...
        }
        *a = 0; *b = 0;                                             // just to make clang-tidy happy...
    }

};

#endif //RAYTRACING_CUDA_UNSTRUCTURED_CONTENT_H
