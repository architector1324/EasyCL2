#ifndef _EASYCL_H_
#define _EASYCL_H_

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define CL_TARGET_OPENCL_VERSION 200
#include "CL/cl.h"

#define ECL_MAX_PLATFORMS_COUNT 32
#define ECL_MAX_DEVICES_COUNT 32

#define ECL_MAX_WORKITEMS_DIMENSION 16

#define ECL_MAX_STRING_LEN 512
#define ECL_MAX_MAP_SIZE 32
#define ECL_MAX_ARRAY_SIZE 32

#define ECL_MAX_PROGRAM_LEN 2048

// "_some" means "hidden from user"

/////////////////////////////////////////
//            Definition
/////////////////////////////////////////

typedef enum {
    ECL_ERROR_OK = 0,
    ECL_ERROR_OUT_OF_MEMORY,
    ECL_ERROR_NO_PLATFORMS,
    ECL_ERROR_NO_PLATFORM,
    ECL_ERROR_NO_DEVICES,
    ECL_ERROR_NO_DEVICE,
    ECL_ERROR_DEVICE_NOT_AVAILABLE,
    ECL_ERROR_LOAD_PROGRAM,
    ECL_ERROR_ALLOCATE_BUFFER,
    ECL_ERROR_BUFFER_NOT_SENDED,
    ECL_ERROR_CREATE_PROGRAM,
    ECL_ERROR_BUILD_PROGRAM,
    ECL_ERROR_NO_COMPILER,
    ECL_ERROR_CREATE_KERNEL,
    ECL_ERROR_NO_KERNEL,
} EclError_t;

typedef struct {
    size_t dim;
    size_t sizes[ECL_MAX_WORKITEMS_DIMENSION];
} EclWorkSize_t;

typedef enum {
    ECL_DEVICE_CPU = CL_DEVICE_TYPE_CPU,
    ECL_DEVICE_GPU = CL_DEVICE_TYPE_GPU,
    ECL_DEVICE_ACCEL = CL_DEVICE_TYPE_ACCELERATOR,
} EclDeviceType_t;

typedef struct {
    EclDeviceType_t type;

    char name[ECL_MAX_STRING_LEN];
    char ext[ECL_MAX_STRING_LEN];
    char ocl_ver[ECL_MAX_STRING_LEN];

    size_t cu; // max compute units
    size_t wrkgSize; // max workgroup size

    EclWorkSize_t wrki; // max workitems sizes

    cl_device_id _id;
} EclDevice_t;

typedef struct {
    char name[ECL_MAX_STRING_LEN];
    char ocl_ver[ECL_MAX_STRING_LEN];
    char ext[ECL_MAX_STRING_LEN];

    cl_platform_id _id;

    size_t cpuCount;
    EclDevice_t cpu[ECL_MAX_DEVICES_COUNT];

    size_t gpuCount;
    EclDevice_t gpu[ECL_MAX_DEVICES_COUNT];

    size_t accelCount;
    EclDevice_t accel[ECL_MAX_DEVICES_COUNT];
} EclPlatform_t;

typedef enum {
    ECL_EXEC_SYNC = 0,
    ECL_EXEC_ASYNC
} EclComputerExec_t;

typedef struct {
    const EclDevice_t* dev;
    cl_context _ctx;
    cl_command_queue _queue;
} EclComputer_t;

typedef struct {
    cl_context _ctx;
    cl_program _prog;
} _EclProgramMap_t;

typedef struct {
    size_t _progSize;
    _EclProgramMap_t _prog[ECL_MAX_MAP_SIZE];
    char src[ECL_MAX_PROGRAM_LEN];
} EclProgram_t;

typedef struct {
    cl_program _prog;
    cl_kernel _kern;
} _EclKernelMap_t;

typedef struct {
    size_t _kernSize;
    _EclKernelMap_t _kern[ECL_MAX_MAP_SIZE];
    char name[ECL_MAX_STRING_LEN];
} EclKernel_t;

typedef enum {
    ECL_BUFFER_READ = CL_MEM_READ_ONLY,
    ECL_BUFFER_WRITE = CL_MEM_WRITE_ONLY,
    ECL_BUFFER_READ_WRITE = CL_MEM_READ_WRITE
} EclBufferAccess_t;

typedef struct {
    cl_context _ctx;
    cl_mem _mem;
} _EclBufferMap_t;

typedef struct {
    size_t _bufSize;
    _EclBufferMap_t _buf[ECL_MAX_MAP_SIZE];
    void* data;
    size_t size;
    EclBufferAccess_t access;
} EclBuffer_t;

typedef struct {
    EclProgram_t* prog;
    EclKernel_t* kern;

    EclBuffer_t* args[ECL_MAX_ARRAY_SIZE];
    size_t argsCount;
} EclFrame_t;

EclError_t eclGetPlatformsCount(size_t* out);
EclError_t eclGetPlatform(size_t id, EclPlatform_t* out);

size_t eclGetDevicesCount(EclDeviceType_t type, EclPlatform_t* platform);
EclError_t eclGetDevice(size_t id, EclDeviceType_t type, EclPlatform_t* platform, EclDevice_t** out);

EclError_t eclComputer(size_t devID, EclDeviceType_t type, EclPlatform_t* platform, EclComputer_t* out);
EclError_t eclComputerSend(EclBuffer_t* arg, const EclComputer_t* comp, EclComputerExec_t exec);
EclError_t eclComputerGrid(EclFrame_t* frame, EclWorkSize_t global, EclWorkSize_t local, const EclComputer_t* comp, EclComputerExec_t exec);
EclError_t eclComputerReceive(EclBuffer_t* arg, const EclComputer_t* comp, EclComputerExec_t exec);
EclError_t eclComputerAwait(const EclComputer_t* comp);
EclError_t eclComputerClear(EclComputer_t* comp);

EclError_t eclProgramLoad(const char* filename, EclProgram_t* out);


// additional wrappers
#define out_of_memory_check(e, f)\
e = f;\
if(e == CL_OUT_OF_HOST_MEMORY || e == CL_OUT_OF_RESOURCES) return ECL_ERROR_OUT_OF_MEMORY


/////////////////////////////////////////
//           Implementation
/////////////////////////////////////////

EclError_t eclGetPlatformsCount(size_t* out) {
    size_t count = 0;

    cl_int err;
    out_of_memory_check(err, clGetPlatformIDs(0, NULL, (cl_uint*)&count));

    if(count == 0) return ECL_ERROR_NO_PLATFORMS;

    *out = count;
    return ECL_ERROR_OK;
}

void _eclGetDevicesArrayByType(EclDeviceType_t type, EclPlatform_t* platform, EclDevice_t** out, size_t** outSize) {
    switch(type) {
    case ECL_DEVICE_CPU:
        if(out) *out = platform->cpu;
        if(outSize) *outSize = &platform->cpuCount;
        break;
    case ECL_DEVICE_GPU:
        if(out) *out = platform->gpu;
        if(outSize) *outSize = &platform->gpuCount;
        break;
    case ECL_DEVICE_ACCEL:
        if(out) *out = platform->accel;
        if(outSize) *outSize = &platform->accelCount;
    default:
        break;
    }
}

EclError_t _eclGetDeviceByID(cl_device_id id, EclDevice_t* out) {
    out->_id = id;

    cl_int err;
    out_of_memory_check(err, clGetDeviceInfo(id, CL_DEVICE_NAME, ECL_MAX_STRING_LEN * sizeof(char), out->name, NULL));
    out_of_memory_check(err, clGetDeviceInfo(id, CL_DEVICE_EXTENSIONS, ECL_MAX_STRING_LEN * sizeof(char), out->ext, NULL));
    out_of_memory_check(err, clGetDeviceInfo(id, CL_DEVICE_VERSION, ECL_MAX_STRING_LEN * sizeof(char), out->ocl_ver, NULL));

    out_of_memory_check(err, clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &out->wrkgSize, NULL));
    out_of_memory_check(err, clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(size_t), &out->wrki.dim, NULL));
    out_of_memory_check(err, clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, out->wrki.dim * sizeof(size_t), out->wrki.sizes, NULL));

    out_of_memory_check(err, clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &out->cu, NULL));

    return ECL_ERROR_OK;
}

EclError_t _eclGetDevicesByType(EclDeviceType_t type, EclPlatform_t* platform) {
    EclDevice_t* out = NULL;
    size_t* outSize = NULL;
    _eclGetDevicesArrayByType(type, platform, &out, &outSize);

    // get devices count
    size_t count = 0;
    
    cl_int err;
    out_of_memory_check(err, clGetDeviceIDs(platform->_id, (cl_device_type)type, 0, NULL, (cl_uint*)&count));

    if(count == 0) return ECL_ERROR_NO_DEVICES;

    // get devices id
    cl_device_id tmp[ECL_MAX_DEVICES_COUNT];
    out_of_memory_check(err, clGetDeviceIDs(platform->_id, (cl_device_type)type, count, tmp, NULL));

    // get devices
    *outSize = count;
    for(size_t i = 0; i < count; i++) {
        EclError_t e = _eclGetDeviceByID(tmp[i], &out[i]);
        if(e != ECL_ERROR_OK) return e;
    }

    return ECL_ERROR_OK;
}

EclError_t eclGetPlatform(size_t id, EclPlatform_t* out) {
    // get platforms count
    size_t count = 0;

    EclError_t err = eclGetPlatformsCount(&count);
    if(err != ECL_ERROR_OK) return err;

    if(id >= count) return ECL_ERROR_NO_PLATFORM;

    // get platform id
    cl_platform_id tmp[ECL_MAX_PLATFORMS_COUNT];

    cl_int tmpErr;
    out_of_memory_check(tmpErr, clGetPlatformIDs(count, tmp, NULL));

    out->_id = tmp[id];

    // get platform name
    out_of_memory_check(tmpErr, clGetPlatformInfo(out->_id, CL_PLATFORM_NAME, ECL_MAX_STRING_LEN * sizeof(char), out->name, NULL));

    // get platform opencl version
    out_of_memory_check(tmpErr, clGetPlatformInfo(out->_id, CL_PLATFORM_VERSION, ECL_MAX_STRING_LEN * sizeof(char), out->ocl_ver, NULL));

    // get platform extensions
    out_of_memory_check(tmpErr, clGetPlatformInfo(out->_id, CL_PLATFORM_EXTENSIONS, ECL_MAX_STRING_LEN * sizeof(char), out->ext, NULL));

    // get platform devices
    err = _eclGetDevicesByType(ECL_DEVICE_CPU, out);
    if(err != ECL_ERROR_OK && err != ECL_ERROR_NO_DEVICES) return err;

    err = _eclGetDevicesByType(ECL_DEVICE_GPU, out);
    if(err != ECL_ERROR_OK && err != ECL_ERROR_NO_DEVICES) return err;

    err = _eclGetDevicesByType(ECL_DEVICE_ACCEL, out);
    if(err != ECL_ERROR_OK && err != ECL_ERROR_NO_DEVICES) return err;

    return ECL_ERROR_OK;
}

size_t eclGetDevicesCount(EclDeviceType_t type, EclPlatform_t* platform) {
    size_t* count = NULL;
    _eclGetDevicesArrayByType(type, platform, NULL, &count);

    return *count;
}

EclError_t eclGetDevice(size_t id, EclDeviceType_t type, EclPlatform_t* platform, EclDevice_t** out) {
    size_t* count = NULL;
    EclDevice_t* devices = NULL;
    _eclGetDevicesArrayByType(type, platform, &devices, &count);

    if(id >= *count) return ECL_ERROR_NO_DEVICE;
    *out = &devices[id];

    return ECL_ERROR_OK;
}

EclError_t eclComputer(size_t devID, EclDeviceType_t type, EclPlatform_t* platform, EclComputer_t* out) {
    // get device
    EclDevice_t* dev = NULL;

    EclError_t err = eclGetDevice(devID, type, platform, &dev);
    if(err != ECL_ERROR_OK) return err;

    out->dev = dev;

    // create context and queue
    cl_int tmpErr;
    out->_ctx = clCreateContext(NULL, 1, &out->dev->_id, NULL, NULL, &tmpErr);
    if(tmpErr == CL_DEVICE_NOT_AVAILABLE) return ECL_ERROR_DEVICE_NOT_AVAILABLE;
    if(tmpErr == CL_OUT_OF_HOST_MEMORY || tmpErr == CL_OUT_OF_RESOURCES) return ECL_ERROR_OUT_OF_MEMORY;

    out->_queue = clCreateCommandQueueWithProperties(out->_ctx, out->dev->_id, NULL, &tmpErr);
    if(tmpErr == CL_OUT_OF_HOST_MEMORY || tmpErr == CL_OUT_OF_RESOURCES) return ECL_ERROR_OUT_OF_MEMORY;

    return ECL_ERROR_OK;
}

bool _eclCheckBuffer(EclBuffer_t* arg, const EclComputer_t* comp, cl_mem* out) {
    for(size_t i = 0; i < arg->_bufSize; i++) {
        if(arg->_buf[i]._ctx == comp->_ctx) {
            if(out) *out = arg->_buf[i]._mem;
            return true;
        }
    }
    return false;
}

bool _eclCheckProgram(EclProgram_t* prog, const EclComputer_t* comp, cl_program* out) {
    for(size_t i = 0; i < prog->_progSize; i++) {
        if(prog->_prog[i]._ctx == comp->_ctx) {
            if(out) *out = prog->_prog[i]._prog;
            return true;
        }
    }
    return false;
}

bool _eclCheckKernel(EclKernel_t* kern, cl_program prog, cl_kernel* out) {
    for(size_t i = 0; i < kern->_kernSize; i++) {
        if(kern->_kern[i]._prog == prog) {
            if(out) *out = kern->_kern[i]._kern;
            return true;
        }
    }
    return false;
}

EclError_t _eclCreateBuffer(EclBuffer_t* arg, const EclComputer_t* comp, cl_mem* out) {
    // check buffer
    if(_eclCheckBuffer(arg, comp, out)) return ECL_ERROR_OK;

    // create buffer
    if(arg->_bufSize >= ECL_MAX_MAP_SIZE) return ECL_ERROR_ALLOCATE_BUFFER;

    cl_int err;
    _EclBufferMap_t* e = &arg->_buf[arg->_bufSize++];
    e->_ctx = comp->_ctx;
    e->_mem = clCreateBuffer(comp->_ctx, (cl_mem_flags)arg->access, arg->size, NULL, &err);

    *out = e->_mem;

    if(err == CL_OUT_OF_HOST_MEMORY || err == CL_OUT_OF_RESOURCES) return ECL_ERROR_OUT_OF_MEMORY;
    if(err == CL_INVALID_BUFFER_SIZE || err == CL_MEM_OBJECT_ALLOCATION_FAILURE) return ECL_ERROR_ALLOCATE_BUFFER;

    return ECL_ERROR_OK;
}

EclError_t _eclCreateProgram(EclProgram_t* prog, const EclComputer_t* comp, cl_program* out) {
    // check program
    if(_eclCheckProgram(prog, comp, out)) return ECL_ERROR_OK;

    // create program
    if(prog->_progSize >= ECL_MAX_MAP_SIZE) return ECL_ERROR_CREATE_PROGRAM;

    const char* src = prog->src;
    size_t srcLen = strlen(prog->src);

    cl_int err = 0;

    _EclProgramMap_t* e = &prog->_prog[prog->_progSize++];
    e->_ctx = comp->_ctx;
    e->_prog = clCreateProgramWithSource(comp->_ctx, 1, &src, &srcLen, &err);

    *out = e->_prog;

    if(err == CL_OUT_OF_HOST_MEMORY || err == CL_OUT_OF_RESOURCES) return ECL_ERROR_OUT_OF_MEMORY;

    err = clBuildProgram(e->_prog, 0, NULL, NULL, NULL, NULL);
    if(err == CL_OUT_OF_HOST_MEMORY || err == CL_OUT_OF_RESOURCES) return ECL_ERROR_OUT_OF_MEMORY;
    if(err == CL_COMPILER_NOT_AVAILABLE) return ECL_ERROR_NO_COMPILER;
    if(err == CL_BUILD_PROGRAM_FAILURE) return ECL_ERROR_BUILD_PROGRAM;

    return ECL_ERROR_OK;
}

EclError_t _eclCreateKernel(EclKernel_t* kern, cl_program prog, cl_kernel* out) {
    // check kernel
    if(_eclCheckKernel(kern, prog, out)) return ECL_ERROR_OK;

    // create kernel
    if(kern->_kernSize >= ECL_MAX_MAP_SIZE) return ECL_ERROR_CREATE_KERNEL;

    cl_int err = 0;

    _EclKernelMap_t* e = &kern->_kern[kern->_kernSize++];
    e->_prog = prog;
    e->_kern = clCreateKernel(prog, kern->name, &err);

    *out = e->_kern;

    if(err == CL_OUT_OF_HOST_MEMORY || err == CL_OUT_OF_RESOURCES) return ECL_ERROR_OUT_OF_MEMORY;
    if(err == CL_INVALID_KERNEL_NAME) return ECL_ERROR_NO_KERNEL; 

    return ECL_ERROR_OK;
}

EclError_t eclComputerSend(EclBuffer_t* arg, const EclComputer_t* comp, EclComputerExec_t exec) {
    cl_mem mem = 0;
    EclError_t err = _eclCreateBuffer(arg, comp, &mem);
    if(err != ECL_ERROR_OK) return err;

    int16_t tmpErr = clEnqueueWriteBuffer(comp->_queue, mem, CL_FALSE, 0, arg->size, arg->data, 0, NULL, NULL);
    if(tmpErr == CL_MEM_OBJECT_ALLOCATION_FAILURE) return ECL_ERROR_ALLOCATE_BUFFER;

    if(exec == ECL_EXEC_SYNC) {
        err = eclComputerAwait(comp);
        if(err != ECL_ERROR_OK) return err;
    }
    return ECL_ERROR_OK;
}

EclError_t eclComputerGrid(EclFrame_t* frame, EclWorkSize_t global, EclWorkSize_t local, const EclComputer_t* comp, EclComputerExec_t exec) {
    // check program
    cl_program prog = 0;
    EclError_t err = _eclCreateProgram(frame->prog, comp, &prog);
    if(err != ECL_ERROR_OK) return err;

    // check kernel
    cl_kernel kern = 0;
    err = _eclCreateKernel(frame->kern, prog, &kern);
    if(err != ECL_ERROR_OK) return err;

    // set args
    cl_int tmpErr = 0;

    for(size_t i = 0; i < frame->argsCount; i++) {
        cl_mem mem = 0;
        if(!_eclCheckBuffer(frame->args[i], comp, &mem)) return ECL_ERROR_BUFFER_NOT_SENDED;

        tmpErr = clSetKernelArg(kern, i, sizeof(cl_mem), &mem);
        if(tmpErr == CL_OUT_OF_HOST_MEMORY || tmpErr == CL_OUT_OF_RESOURCES) return ECL_ERROR_OUT_OF_MEMORY;
    }

    tmpErr = clEnqueueNDRangeKernel(comp->_queue, kern, global.dim, NULL, global.sizes, local.sizes, 0, NULL, NULL);

    if(exec == ECL_EXEC_SYNC) {
        err = eclComputerAwait(comp);
        if(err != ECL_ERROR_OK) return err;
    }

    return ECL_ERROR_OK;
}

EclError_t eclComputerReceive(EclBuffer_t* arg, const EclComputer_t* comp, EclComputerExec_t exec) {
    cl_mem mem = 0;
    if(!_eclCheckBuffer(arg, comp, &mem)) return ECL_ERROR_BUFFER_NOT_SENDED;

    cl_int err;
    out_of_memory_check(err, clEnqueueReadBuffer(comp->_queue, mem, CL_FALSE, 0, arg->size, arg->data, 0, NULL, NULL));

    if(exec == ECL_EXEC_SYNC) {
        EclError_t tmpErr = eclComputerAwait(comp);
        if(tmpErr != ECL_ERROR_OK) return tmpErr;
    }

    return ECL_ERROR_OK;
}

EclError_t eclComputerAwait(const EclComputer_t* comp) {
    cl_int err;
    out_of_memory_check(err, clFinish(comp->_queue));

    return ECL_ERROR_OK;
}

EclError_t eclComputerClear(EclComputer_t* comp) {
    cl_int err;
    out_of_memory_check(err, clReleaseContext(comp->_ctx));
    out_of_memory_check(err, clReleaseCommandQueue(comp->_queue));

    comp->_ctx = 0;
    comp->_queue = 0;
    comp->dev = NULL;

    return ECL_ERROR_OK;
}

EclError_t eclProgramLoad(const char* name, EclProgram_t* out) {
    size_t i = 0;
    FILE* f = fopen(name, "r");

    if(!f) return ECL_ERROR_LOAD_PROGRAM;

    while(!feof(f)) {
        if(i >= ECL_MAX_PROGRAM_LEN) {
            fclose(f);
            return ECL_ERROR_LOAD_PROGRAM;
        }
        out->src[i++] = fgetc(f);
    }
    out->src[--i] = '\0';
    fclose(f);

    return ECL_ERROR_OK;
}


#endif // _EASY_CL_H_
