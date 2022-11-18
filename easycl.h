#ifndef _EASYCL_H_
#define _EASYCL_H_

#include <stddef.h>
#include <stdint.h>

#define CL_TARGET_OPENCL_VERSION 210
#include "CL/cl.h"

#define ECL_MAX_PLATFORMS_COUNT 32
#define ECL_MAX_DEVICES_COUNT 32
#define ECL_MAX_STRING_LEN 512

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
    ECL_ERROR_NO_DEVICE
} EclError_t;

typedef enum {
    ECL_DEVICE_CPU = CL_DEVICE_TYPE_CPU,
    ECL_DEVICE_GPU = CL_DEVICE_TYPE_GPU,
    ECL_DEVICE_ACCEL = CL_DEVICE_TYPE_ACCELERATOR,
} EclDeviceType_t;

typedef struct {
    char name[ECL_MAX_STRING_LEN];
    char ext[ECL_MAX_STRING_LEN];
    EclDeviceType_t type;

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

EclError_t eclGetPlatformsCount(size_t* out);
EclError_t eclGetPlatform(size_t id, EclPlatform_t* out);

size_t eclGetDevicesCount(EclDeviceType_t type, EclPlatform_t* platform);
EclError_t eclGetDevice(size_t id, EclDeviceType_t type, EclPlatform_t* platform, EclDevice_t** out);

/////////////////////////////////////////
//           Implementation
/////////////////////////////////////////

EclError_t eclGetPlatformsCount(size_t* out) {
    size_t count = 0;

    cl_int err = clGetPlatformIDs(0, NULL, (cl_uint*)&count);
    if(err == CL_OUT_OF_HOST_MEMORY) return ECL_ERROR_OUT_OF_MEMORY;

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

    cl_int err = clGetDeviceInfo(id, CL_DEVICE_NAME, ECL_MAX_STRING_LEN * sizeof(char), out->name, NULL);
    if(err == CL_OUT_OF_HOST_MEMORY) return ECL_ERROR_OUT_OF_MEMORY;

    err = clGetDeviceInfo(id, CL_DEVICE_EXTENSIONS, ECL_MAX_STRING_LEN * sizeof(char), out->ext, NULL);
    if(err == CL_OUT_OF_HOST_MEMORY) return ECL_ERROR_OUT_OF_MEMORY;

    return ECL_ERROR_OK;
}

EclError_t _eclGetDevicesByType(EclDeviceType_t type, EclPlatform_t* platform) {
    EclDevice_t* out;
    size_t* outSize;
    _eclGetDevicesArrayByType(type, platform, &out, &outSize);

    // get devices count
    size_t count = 0;
    
    cl_int err = clGetDeviceIDs(platform->_id, (cl_device_type)type, 0, NULL, (cl_uint*)&count);
    if(err == CL_OUT_OF_HOST_MEMORY || err == CL_OUT_OF_RESOURCES) return ECL_ERROR_OUT_OF_MEMORY;

    if(count == 0) return ECL_ERROR_NO_DEVICES;

    // get devices id
    cl_device_id tmp[ECL_MAX_DEVICES_COUNT];
    err = clGetDeviceIDs(platform->_id, (cl_device_type)type, count, tmp, NULL);

    if(err == CL_OUT_OF_HOST_MEMORY || err == CL_OUT_OF_RESOURCES) return ECL_ERROR_OUT_OF_MEMORY;

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
    cl_uint tmpErr = clGetPlatformIDs(count, tmp, NULL);
    if(tmpErr == CL_OUT_OF_HOST_MEMORY) return ECL_ERROR_OUT_OF_MEMORY;

    out->_id = tmp[id];

    // get platform name
    tmpErr = clGetPlatformInfo(out->_id, CL_PLATFORM_NAME, ECL_MAX_STRING_LEN * sizeof(char), out->name, NULL);
    if(tmpErr == CL_OUT_OF_HOST_MEMORY) return ECL_ERROR_OUT_OF_MEMORY;

    // get platform opencl version
    tmpErr = clGetPlatformInfo(out->_id, CL_PLATFORM_VERSION, ECL_MAX_STRING_LEN * sizeof(char), out->ocl_ver, NULL);
    if(tmpErr == CL_OUT_OF_HOST_MEMORY) return ECL_ERROR_OUT_OF_MEMORY;

    // get platform extensions
    tmpErr = clGetPlatformInfo(out->_id, CL_PLATFORM_EXTENSIONS, ECL_MAX_STRING_LEN * sizeof(char), out->ext, NULL);
    if(tmpErr == CL_OUT_OF_HOST_MEMORY) return ECL_ERROR_OUT_OF_MEMORY;

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
    size_t* count;
    _eclGetDevicesArrayByType(type, platform, NULL, &count);

    return *count;
}

EclError_t eclGetDevice(size_t id, EclDeviceType_t type, EclPlatform_t* platform, EclDevice_t** out) {
    size_t* count;
    EclDevice_t* devices;
    _eclGetDevicesArrayByType(type, platform, &devices, &count);

    if(id >= *count) return ECL_ERROR_NO_DEVICE;
    *out = &devices[id];

    return ECL_ERROR_OK;
}

#endif // _EASY_CL_H_
