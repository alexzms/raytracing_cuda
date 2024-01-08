//
// Created by alexzms on 2024/1/4.
//

#ifndef RAYTRACING_CUDA_CAMERA_H
#define RAYTRACING_CUDA_CAMERA_H

#include "vec3.h"

namespace rt_cuda {
    namespace kernel_funcs {
        __global__ void render() {

        }
    }

    class camera {
    public:
        camera() = default;


    private:
        GLsizei _width;
        GLsizei _height;
        uchar3* _d_ptr;
        size_t _d_ptr_size;

    };
}

#endif //RAYTRACING_CUDA_CAMERA_H
