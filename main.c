#include <stdio.h>
#include <string.h>
#include "easycl.h"


int main() {
    EclPlatform_t plat = {};
    eclGetPlatform(0, &plat);

    EclDevice_t* gpu = NULL;
    eclGetDevice(0, ECL_DEVICE_GPU, &plat, &gpu);

    printf("Platform: {name:`%s` oclVer:`%s` ext:`%s`}\n", plat.name, plat.ocl_ver, plat.ext);

    char tmp[ECL_MAX_STRING_LEN];
    for(size_t i = 0; i < gpu->wrkiDim; i++) {
        char tmp2[ECL_MAX_STRING_LEN];
        sprintf(tmp2, "%lu%s", gpu->wrkiSizes[i], i < gpu->wrkiDim - 1 ? ", " : "");
        strncat(tmp, tmp2, 16);
    }
    printf("Device info: {name:`%s` oclVer:`%s` cu:%lu wrkgSize:%lu wrkiDim:%lu wrkiSizes:[%s]}\n", gpu->name, gpu->ocl_ver, gpu->cu, gpu->wrkgSize, gpu->wrkiDim, tmp);

    return 0;
}
