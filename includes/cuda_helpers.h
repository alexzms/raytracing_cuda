//
// Created by alexzms on 2024/1/1.
//

#ifndef RAYTRACING_CUDA_CUDA_HELPERS_H
#define RAYTRACING_CUDA_CUDA_HELPERS_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#endif //RAYTRACING_CUDA_CUDA_HELPERS_H
