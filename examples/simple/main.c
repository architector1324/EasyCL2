#include <stdio.h>
#include <string.h>
#include "easycl.h"


int main() {
    // setup program and kernel
    EclProgram_t prog = {};
    eclProgramLoad("main.cl", &prog);

    EclKernel_t kern = {.name = "test"};

    // get platform, setup the computer
    EclPlatform_t plat = {};
    eclGetPlatform(0, &plat);

    EclComputer_t gpu = {};
    eclComputer(0, ECL_DEVICE_GPU, &plat, &gpu);

    // setup data container
    int data[12] = {};

    EclBuffer_t a = {
        .data = data,
        .size = 12 * sizeof(float),
        .access = ECL_BUFFER_READ_WRITE
    };

    // setup compute frame
    EclFrame_t frame = {
        .prog = &prog,
        .kern = &kern,
        .args = {&a},
        .argsCount = 1
    };

    // 12 compute units combined into 3 groups
    eclComputerSend(&a, &gpu, ECL_EXEC_SYNC);
    eclComputerGrid(&frame, (EclWorkSize_t){.dim = 1, .sizes={12}}, (EclWorkSize_t){.dim = 1, .sizes={3}}, &gpu, ECL_EXEC_SYNC);
    eclComputerReceive(&a, &gpu, ECL_EXEC_SYNC);

    // output
    for(size_t i = 0; i < 12; i++)
        printf("%d\n", data[i]);

    // clean resources
    eclBufferClear(&a);

    eclComputerClear(&gpu);
    eclPlatformClear(&plat);

    eclKernelClear(&kern);
    eclProgramClear(&prog);

    return 0;
}
