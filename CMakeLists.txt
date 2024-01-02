cmake_minimum_required(VERSION 3.27)
project(raytracing_cuda LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)


#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_52)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_70,code=sm_70)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=sm_75)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_80,code=sm_80)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_86,code=sm_86)

set_source_files_properties(
        src/main.cu
        includes/cuda_helpers.h
        includes/unstructured_content.h
        PROPERTIES
        LANGUAGE CUDA
)

set_source_files_properties(
        lib/glad.c
        PROPERTIES
        LANGUAGE CXX
)

include_directories(E:/ComputerGraphics/libraries/OpenGL/includes)
include_directories(E:/ComputerGraphics/libraries/Utilities/includes)
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/include")
include_directories(E:/ComputerGraphics/libraries/Visualime/includes)
include_directories(includes)
link_directories(E:/ComputerGraphics/libraries/OpenGL/lib)
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64")
link_directories(E:/ComputerGraphics/libraries/Visualime/lib)


add_executable(raytracing_cuda
        src/main.cu
        lib/glad.c
)

target_link_libraries(raytracing_cuda PUBLIC glfw3.dll opengl32.lib visualime.lib)
target_link_libraries(raytracing_cuda PUBLIC
        ${CUDA_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
        cudart
)

set_target_properties(raytracing_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)