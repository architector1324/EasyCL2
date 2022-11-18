#include <stdio.h>
#include "easycl.h"


int main() {
    EclPlatform_t plat = {};
    eclGetPlatform(0, &plat);

    EclDevice_t* gpu = NULL;
    eclGetDevice(0, ECL_DEVICE_GPU, &plat, &gpu);

    printf("Platform: %s %s %s\n", plat.name, plat.ocl_ver, plat.ext);
    printf("Device: %s %s\n", gpu->name, gpu->ext);

    return 0;
}
