#include <iostream>
#ifndef VISUALIME_USE_CUDA
#define VISUALIME_USE_CUDA
#endif
#include "visualime/visualime.h"
#include "raytracing_cuda.h"

void canvas_scene_cuda_2d_test() {
    using namespace visualime;

    scene::canvas_scene_cuda_2d scene{800, 800};
    scene.launch(true, 60);

    int incremental_color = 0;
    while (scene.is_running()) {
        CHECK_ERROR(cudaMemset(scene.get_d_ptr(), incremental_color, scene.get_d_ptr_size()));
        incremental_color += 1;
        if (incremental_color > 255) {
            incremental_color = 0;
        }
        scene.refresh();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (scene.get_run_thread().joinable()) {
        scene.get_run_thread().join();
    }

    std::cout << "Test canvas_scene_cuda_2d finished" << std::endl;
}
void output_an_image_test() {
    GLsizei coeff = 1;
    GLsizei width = 2560, height = 1601;
    visualime::scene::canvas_scene_cuda_2d scene{width, height, (double)coeff, true};
    scene.launch(true, 60);
    scene.wait_for_running();
    rt_cuda::output_an_image OAI{2560 * coeff, 1600 * coeff, scene.get_d_ptr(), scene.get_d_ptr_size()};

    scene.refresh();
    unsigned char z = 0;
    int sign = 1;
    while(scene.is_running()) {
        OAI.run(z);
        scene.refresh();
        if (z == 255) sign = -1;
        if (z == 0) sign = 1;
        z += 3 * sign;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (scene.get_run_thread().joinable()) {
        scene.get_run_thread().join();
    }
}


int main() {
    std::cout << "Hello, World!" << std::endl;
//    canvas_scene_cuda_2d_test();
    output_an_image_test();
    return 0;
}
