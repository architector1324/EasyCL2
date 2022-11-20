#include <stdio.h>
#include <string.h>
#include "easycl.h"


int main() {
    // setup program and kernel
    // EclProgram_t prog = {};
    // eclProgramLoad("main.cl", &prog);

    // EclKernel_t kern = {.name = "test"};

    // get platform, setup the computer
    EclPlatform_t plat = {};
    eclGetPlatform(0, &plat);

    EclComputer_t gpu = {};
    eclComputer(0, ECL_DEVICE_GPU, &plat, &gpu);

    // setup data container
    float data[12] = {1.0f, 2.0f, 3.0f};
    float data2[12] = {};

    EclBuffer_t a = {
        .data = data,
        .size = 12 * sizeof(float),
        .access = ECL_BUFFER_READ
    };

    // // setup compute frame
    // EclFrame_t frame = {
    //     .prog = &prog,
    //     .kern = &kern,
    //     .args = {&a},
    //     .argsCount = 1
    // };

    // 12 compute units combined into 3 groups
    eclComputerSend(&a, &gpu, ECL_EXEC_SYNC);
    a.data = data2;

    eclComputerReceive(&a, &gpu, ECL_EXEC_SYNC);

    // clean resources
    eclComputerClear(&gpu);

    return 0;
}
