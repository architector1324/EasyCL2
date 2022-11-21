# Easy Computing Library v2


## Overview
***EasyCL v2*** is open-source (*GPL v3*) **header-only** wrapper above [***OpenCL***](https://www.khronos.org/opencl/) library. It's designed for easy and convenient use, allowing you to quickly write the host part of the program and flexibly use OpenCL kernels.

The library is built on the original [***OpenCL 2.0 C Specification***](https://www.khronos.org/registry/OpenCL/specs/opencl-2.0.pdf). It allows you to bypass some of the inconveniences of the original *OpenCL*, providing work with **abstractions** of different levels, provides you to bypass the **restriction of hard binding**.

[First](https://github.com/architector1324/EasyCL) version of this library was made for **C++**.

## Installation
 1) Install OpenCL library on your system.
 2) Clone the repo `$ git clone https://github.com/architector1324/EasyCL2`.
 3) Copy `easycl.h` to your project

## Hello, World
 1) Copy `easycl.h` to project folder
 2) Create `main.c`:

```c
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
```

 3) Create `main.cl`:
```c
kernel void test(global int* a, int b){
    size_t i = get_global_id(0);
    a[i] = b * ((int)get_group_id(0) + 1);
}
```

 4) Type in terminal:
```bash
$ gcc -O3 -lOpenCL main.c -o a.out
$ ./a.out
```

Output:
```
5
5
5
10
10
10
15
15
15
20
20
20
```

If you have any questions, feel free to contact me olegsajaxov@yandex.ru
