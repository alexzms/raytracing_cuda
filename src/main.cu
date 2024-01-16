#include <iostream>
#ifndef VISUALIME_USE_CUDA
#define VISUALIME_USE_CUDA
#endif
#include "visualime/visualime.h"
#include "raytracing_cuda.h"
//#undef CHECK_ERROR
//#define CHECK_ERROR(x) x


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

__global__ void create_assets(rt_cuda::texture_base** d_textures, size_t num_texture,
                              rt_cuda::material_base** d_materials, size_t num_material,
                              rt_cuda::hittable** d_hittables, size_t num_hittable,
                              rt_cuda::hittable_list** world)
{
    using namespace rt_cuda;
    d_textures[0] = new solid_color{color3f{0.2, 0.7, 0.6}};
    d_textures[1] = new solid_color{color3f{0.4, 0.1, 0.9}};
    d_materials[0] = new lambertian{d_textures[0]};
    d_materials[1] = new lambertian{d_textures[1]};
    d_hittables[0] = new sphere{point3f{0, 3, 0}, 1.0f, d_materials[0]};
    d_hittables[1] = new sphere{point3f{0, -100, 0}, 102.0f, d_materials[1]};

    (*world) = new hittable_list{d_hittables, num_hittable};
}

__global__ void create_camera(rt_cuda::camera** cam, GLsizei width, GLsizei height, unsigned spp) {
    using namespace rt_cuda;
    (*cam) = new camera{};
    (*cam)->set_camera_parameter(width, height);
    (*cam)->set_focus_parameter(0);
    (*cam)->vfov     = 40;
    (*cam)->lookfrom = point3f(10, 10, -10);
    (*cam)->lookat   = point3f(0, 0, 0);
    (*cam)->vup      = vec3f(0,1,0);
    (*cam)->samples_per_pixel = spp;
    (*cam)->initialize();
}

__global__ void destroy_assets(rt_cuda::texture_base** d_textures, size_t num_texture,
                               rt_cuda::material_base** d_materials, size_t num_material,
                               rt_cuda::hittable** d_hittables, size_t num_hittable,
                               rt_cuda::hittable_list** d_world, rt_cuda::camera** d_cam)
{
//    for (size_t i = 0; i != num_texture; ++i) {
//        delete d_textures[i];
//    }
//    for (size_t i = 0; i != num_material; ++i) {
//        delete d_materials[i];
//    }
//    for (size_t i = 0; i != num_hittable; ++i) {
//        delete d_hittables[i];
//    }
//    delete *d_world;
//    delete *d_cam;
}


void background_color_scene() {
    using namespace rt_cuda;
    size_t num_texture = 2, num_material = 2, num_hittable = 2;
    // malloc all memory space for kernel func: create_assets
    texture_base** d_textures, **h_textures;
    material_base** d_materials, **h_materials;
    hittable** d_hittables, **h_hittables;
    hittable_list** d_world, **h_world;
    h_textures = (texture_base**)malloc(sizeof(texture_base*) * num_texture);
    h_materials = (material_base**)malloc(sizeof(material_base*) * num_material);
    h_hittables = (hittable**)malloc(sizeof(hittable*) * num_hittable);
    h_world = (hittable_list**)malloc(sizeof(hittable_list*));
    CHECK_ERROR(cudaMalloc((void**)&d_textures, sizeof(texture_base*) * num_texture));
    CHECK_ERROR(cudaMalloc((void**)&d_materials, sizeof(material_base*) * num_material));
    CHECK_ERROR(cudaMalloc((void**)&d_hittables, sizeof(hittable*) * num_material));
    CHECK_ERROR(cudaMalloc((void**)&d_world, sizeof(hittable_list*)));
    // create assets, copy assets memory, now the memory model is like
    //  [for textures, materials, hittables]
    //      d_textures-> |(d_address0) address of d_texture 0 |(d_address1) address of d_texture 1 |...
    //      | copy   |
    //      v        v
    //      h_textures-> |(h_address0) address of d_texture 0 |(h_address1) address of d_texture 1 |...
    //  [for world]
    //      d_world ->   |(d_address0) address of d_world -> memory of d_world
    //      |     |                                            ^
    //      v     v                                            |
    //      h_world->    |(h_address0) address of d_world _____|
    //      which is just what I want
    create_assets<<<1, 1>>>(d_textures, num_texture, d_materials, num_material, d_hittables, num_hittable, d_world);
    CHECK_ERROR(cudaDeviceSynchronize());
    CHECK_ERROR(cudaMemcpy(h_textures, d_textures, sizeof(texture_base*) * num_texture, cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(h_materials, d_materials, sizeof(material_base*) * num_material, cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(h_hittables, d_hittables, sizeof(hittable*) * num_hittable, cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(h_world, d_world, sizeof(hittable_list*), cudaMemcpyDeviceToHost));
//    textures_wrapper textures;
//    materials_wrapper materials;
//    textures.store_inside(h_textures, num_texture);
//    materials.store_inside(h_materials, num_material);
    // now we can free the array tha stores all the *d_textures on hosts, which is no longer needed
    // notice: h_world** should not be freed now
    free(h_textures); free(h_materials); free(h_hittables);

    GLsizei coeff = 1;
    GLsizei width = 2560, height = 1600;
    camera** h_cam, **d_cam;
    h_cam = (camera**)malloc(sizeof(camera*));
    CHECK_ERROR(cudaMalloc((void**)&d_cam, sizeof(camera*)));
    create_camera<<<1, 1>>>(d_cam, width, height, 1);
    CHECK_ERROR(cudaDeviceSynchronize());
    CHECK_ERROR(cudaMemcpy(h_cam, d_cam, sizeof(camera*), cudaMemcpyDeviceToHost));

    visualime::scene::canvas_scene_cuda_2d scene{width, height, (double)coeff, true};
    scene.set_constant_refresh(true);
    scene.launch(true, 60);
    scene.wait_for_running();
    camera_wrapper cam_wrapper{*h_cam, width, height, scene.get_d_ptr(), 1};
    while (scene.is_running()) {
        cudaEvent_t start, stop;
        CHECK_ERROR(cudaEventCreate(&start));
        CHECK_ERROR(cudaEventCreate(&stop));
        CHECK_ERROR(cudaEventRecord(start, nullptr));
        cam_wrapper.accumu_render(*h_world);
        CHECK_ERROR(cudaDeviceSynchronize());
        CHECK_ERROR(cudaEventRecord(stop, nullptr));
        CHECK_ERROR(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CHECK_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("[camera][info] render time: %f ms\n", milliseconds);
//        scene.refresh();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (scene.get_run_thread().joinable()) {
        scene.get_run_thread().join();
    }
    destroy_assets<<<1, 1>>>(d_textures, num_texture, d_materials, num_material, d_hittables, num_hittable, d_world, d_cam);
    CHECK_ERROR(cudaDeviceSynchronize());
    CHECK_ERROR(cudaFree(d_textures)); CHECK_ERROR(cudaFree(d_materials));
    CHECK_ERROR(cudaFree(d_hittables)); CHECK_ERROR(cudaFree(d_world));
    CHECK_ERROR(cudaFree(d_cam));

    free(h_world); free(h_cam);
}


int main() {
    std::cout << "Hello, World!" << std::endl;
//    canvas_scene_cuda_2d_test();
//    output_an_image_test();
    background_color_scene();
    return 0;
}
