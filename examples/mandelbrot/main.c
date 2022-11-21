#include <stdio.h>
#include <string.h>
#include "easycl.h"


void save_ppm(const char* filename, const uint8_t* data, size_t w, size_t h) {
    FILE* f = fopen(filename, "w");
    fprintf(f, "P3\n%lu %lu\n255\n", w, h);

    for(size_t i = 0; i < w * h; i++) {
        fprintf(f, "%u %u %u\n", data[3 * i], data[3 * i + 1], data[3 * i + 2]);
    }
    fclose(f);
}

void mandelbrot_cpu(uint8_t* data, uint32_t w, uint32_t h, float px, float py, float mag, uint32_t maxIter) {
    float aspect = w / h;

    for(int x = 0; x < w; x++) {
        for(int y = 0; y < h; y++) {
            float i = ((float)x - w / 2) / (mag * w / 4) - px;
            float j = ((float)y - h / 2) / (mag * h * aspect / 4) - py;

            float oldI = i;
            float oldJ = j;

            uint32_t k = 0;

            for(; k < maxIter; k++) {
                float a = i * i - j * j;
                float b = 2 * i * j;
                i = a + oldI;
                j = b + oldJ;

                if(i * i + j * j > 4) break;
            }

            uint32_t value = 255 * k / maxIter;

            data[3 * (x + w * y)] = value;
            data[3 * (x + w * y) + 1] = value;
            data[3 * (x + w * y) + 2] = value;
        }
    }
}

int main() {
    uint32_t w = 8192;
    uint32_t h = 4096;

    uint8_t* data = malloc(w * h * 3 * sizeof(uint8_t));

    float px = 0.65f;
    float py = 0;
    float mag = 1.0f;
    uint32_t maxIter = 100;

    // cpu
    mandelbrot_cpu(data, w, h, px, py, mag, maxIter);
    save_ppm("out_cpu.ppm", data, w, h);

    memset(data, 0, w * h * 3 * sizeof(uint8_t));

    // gpu
    // setup program and kernel
    EclProgram_t prog = {};
    eclProgramLoad("main.cl", &prog);

    EclKernel_t kern = {.name = "mandelbrot"};

    // get platform, setup the computer
    EclPlatform_t plat = {};
    eclGetPlatform(0, &plat);

    EclComputer_t gpu = {};
    eclComputer(0, ECL_DEVICE_GPU, &plat, &gpu);

    // setup data container
    EclBuffer_t dataBuf = {
        .data = data,
        .size = w * h * 3 * sizeof(uint8_t),
        .access = ECL_BUFFER_WRITE
    };

    // setup compute frame
    EclFrame_t frame = {
        .prog = &prog,
        .kern = &kern,
        .args = {
            {ECL_ARG_BUFFER, &dataBuf},
            {ECL_ARG_VAR, &w, sizeof(uint32_t)},
            {ECL_ARG_VAR, &h, sizeof(uint32_t)},
            {ECL_ARG_VAR, &px, sizeof(float)},
            {ECL_ARG_VAR, &py, sizeof(float)},
            {ECL_ARG_VAR, &mag, sizeof(float)},
            {ECL_ARG_VAR, &maxIter, sizeof(uint32_t)}
        },
        .argsCount = 7
    };

    // compute
    eclComputerSend(&dataBuf, &gpu, ECL_EXEC_SYNC);
    eclComputerGrid(&frame, (EclWorkSize_t){.dim = 2, .sizes={w, h}}, (EclWorkSize_t){.dim = 2, .sizes={256, 1}}, &gpu, ECL_EXEC_SYNC);
    eclComputerReceive(&dataBuf, &gpu, ECL_EXEC_SYNC);

    // output
    save_ppm("out_gpu.ppm", data, w, h);

    // clean resources
    eclBufferClear(&dataBuf);

    eclComputerClear(&gpu);
    eclPlatformClear(&plat);

    eclKernelClear(&kern);
    eclProgramClear(&prog);

    free(data);

    return 0;
}
