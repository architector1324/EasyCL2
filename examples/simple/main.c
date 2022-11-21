#include <stdio.h>
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
    ecl_array(int, a, 12, {}, ECL_BUFFER_WRITE); // ecl::array<int> a[12] = {};
    int b = 5;

    // setup compute frame
    EclFrame_t frame = {
        .prog = &prog,
        .kern = &kern,
        .args = {{ECL_ARG_BUFFER, &a}, {ECL_ARG_VAR, &b, sizeof(int)}},
        .argsCount = 2
    };

    // 12 compute units combined into 3 groups
    eclComputerSend(&a, &gpu, ECL_EXEC_SYNC);
    eclComputerGrid(&frame, (EclWorkSize_t){.dim = 1, .sizes={12}}, (EclWorkSize_t){.dim = 1, .sizes={3}}, &gpu, ECL_EXEC_SYNC);
    eclComputerReceive(&a, &gpu, ECL_EXEC_SYNC);

    // output
    for(size_t i = 0; i < 12; i++)
        printf("%d\n", ((const int*)a.data)[i]);

    // clean resources
    eclBufferClear(&a);

    eclComputerClear(&gpu);
    eclPlatformClear(&plat);

    eclKernelClear(&kern);
    eclProgramClear(&prog);

    return 0;
}
