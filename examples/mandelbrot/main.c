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

void mandelbrot_cpu(uint8_t* data, size_t w, size_t h, size_t maxIter) {
    float aspect = w / h;

    for(int x = 0; x < w; x++) {
        for(int y = 0; y < h; y++) {
            float i = ((float)x - w / 2) / (w / 4) - 0.65f;
            float j = ((float)y - h / 2) / (h * aspect / 4);

            float oldI = i;
            float oldJ = j;

            size_t k = 0;

            for(; k < maxIter; k++) {
                float a = i * i - j * j;
                float b = 2 * i * j;
                i = a + oldI;
                j = b + oldJ;

                if(i * i + j * j > 4) break;
            }

            size_t value = 255 * k / maxIter;

            data[3 * (x + w * y)] = value;
            data[3 * (x + w * y) + 1] = value;
            data[3 * (x + w * y) + 2] = value;
        }
    }
}

int main() {
    size_t w = 2048;
    size_t h = 1024;
    size_t maxIter = 80;

    uint8_t* data = malloc(2048 * 1024 * 3 * sizeof(uint8_t));

    // cpu
    mandelbrot_cpu(data, w, h, maxIter);
    save_ppm("out_cpu.ppm", data, w, h);

    memset(data, 0, 2048 * 1024 * 3 * sizeof(uint8_t));

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
        .access = ECL_BUFFER_READ_WRITE
    };

    EclBuffer_t wData = {
        .data = &w,
        .size = sizeof(size_t),
        .access = ECL_BUFFER_READ
    };

    EclBuffer_t hData = {
        .data = &h,
        .size = sizeof(size_t),
        .access = ECL_BUFFER_READ
    };

    EclBuffer_t maxIterData = {
        .data = &maxIter,
        .size = sizeof(size_t),
        .access = ECL_BUFFER_READ
    };

    // setup compute frame
    EclFrame_t frame = {
        .prog = &prog,
        .kern = &kern,
        .args = {&dataBuf, &wData, &hData, &maxIterData},
        .argsCount = 4
    };

    // compute
    eclComputerSend(&dataBuf, &gpu, ECL_EXEC_SYNC);
    eclComputerSend(&wData, &gpu, ECL_EXEC_SYNC);
    eclComputerSend(&hData, &gpu, ECL_EXEC_SYNC);
    eclComputerSend(&maxIterData, &gpu, ECL_EXEC_SYNC);

    eclComputerGrid(&frame, (EclWorkSize_t){.dim = 2, .sizes={w, h}}, (EclWorkSize_t){.dim = 2, .sizes={256, 1}}, &gpu, ECL_EXEC_SYNC);
    eclComputerReceive(&dataBuf, &gpu, ECL_EXEC_SYNC);

    // output
    save_ppm("out_gpu.ppm", data, w, h);

    // clean resources
    eclComputerClear(&gpu);
    free(data);

    return 0;
}
