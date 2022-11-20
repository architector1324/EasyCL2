#include <stdio.h>
#include <string.h>
#include "easycl.h"


int main() {
    EclPlatform_t plat = {};
    eclGetPlatform(0, &plat);

    EclComputer_t gpu = {};
    eclComputer(0, ECL_DEVICE_GPU, &plat, &gpu);

    EclProgram_t prog = {};
    eclProgramLoad("main.cl", &prog);

    eclReleaseComputer(&gpu);

    return 0;
}
