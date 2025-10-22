/*
 * Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd. All rights reserved.
 *
 * This file is part of Nvtop.
 *
 * Nvtop is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Nvtop is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with nvtop.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "nvtop/common.h"
#include "nvtop/extract_gpuinfo_common.h"

#include <dlfcn.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NVML_SUCCESS 0
#define NVML_ERROR_INSUFFICIENT_SIZE 7

typedef struct nvmlDevice *nvmlDevice_t;
typedef int nvmlReturn_t; // store the enum as int

// Init and shutdown

static nvmlReturn_t (*nvmlInit)(void);

static nvmlReturn_t (*nvmlShutdown)(void);

// Static information and helper functions

static nvmlReturn_t (*nvmlDeviceGetCount)(unsigned int *deviceCount);

static nvmlReturn_t (*nvmlDeviceGetHandleByIndex)(unsigned int index, nvmlDevice_t *device);

static const char *(*nvmlErrorString)(nvmlReturn_t);

static nvmlReturn_t (*nvmlDeviceGetName)(nvmlDevice_t device, char *name, unsigned int length);

typedef struct {
  char busIdLegacy[16];
  unsigned int domain;
  unsigned int bus;
  unsigned int device;
  unsigned int pciDeviceId;
  // Added in NVML 2.285 API
  unsigned int pciSubSystemId;
  char busId[32];
} nvmlPciInfo_t;

static nvmlReturn_t (*nvmlDeviceGetPciInfo)(nvmlDevice_t device, nvmlPciInfo_t *pciInfo);

static nvmlReturn_t (*nvmlDeviceGetMaxPcieLinkGeneration)(nvmlDevice_t device, unsigned int *maxLinkGen);

static nvmlReturn_t (*nvmlDeviceGetMaxPcieLinkWidth)(nvmlDevice_t device, unsigned int *maxLinkWidth);

typedef enum {
  NVML_TEMPERATURE_THRESHOLD_SHUTDOWN = 0,
  NVML_TEMPERATURE_THRESHOLD_SLOWDOWN = 1,
  NVML_TEMPERATURE_THRESHOLD_MEM_MAX = 2,
  NVML_TEMPERATURE_THRESHOLD_GPU_MAX = 3,
  NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN = 4,
  NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR = 5,
  NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX = 6,
} nvmlTemperatureThresholds_t;

static nvmlReturn_t (*nvmlDeviceGetTemperatureThreshold)(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType,
                                                         unsigned int *temp);

// Dynamic information extraction

typedef enum {
  NVML_CLOCK_GRAPHICS = 0,
  NVML_CLOCK_SM = 1,
  NVML_CLOCK_MEM = 2,
  NVML_CLOCK_VIDEO = 3,
} nvmlClockType_t;

static nvmlReturn_t (*nvmlDeviceGetClockInfo)(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);

static nvmlReturn_t (*nvmlDeviceGetMaxClockInfo)(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);

typedef struct {
  unsigned int gpu;
  unsigned int memory;
} nvmlUtilization_t;

static nvmlReturn_t (*nvmlDeviceGetUtilizationRates)(nvmlDevice_t device, nvmlUtilization_t *utilization);

typedef struct {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
} nvmlMemory_v1_t;

typedef struct {
  unsigned int version;
  unsigned long long total;
  unsigned long long reserved;
  unsigned long long free;
  unsigned long long used;
} nvmlMemory_v2_t;

static nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(nvmlDevice_t device, nvmlMemory_v1_t *memory);
static nvmlReturn_t (*nvmlDeviceGetMemoryInfo_v2)(nvmlDevice_t device, nvmlMemory_v2_t *memory);

static nvmlReturn_t (*nvmlDeviceGetCurrPcieLinkGeneration)(nvmlDevice_t device, unsigned int *currLinkGen);

static nvmlReturn_t (*nvmlDeviceGetCurrPcieLinkWidth)(nvmlDevice_t device, unsigned int *currLinkWidth);

typedef enum {
  NVML_PCIE_UTIL_TX_BYTES = 0,
  NVML_PCIE_UTIL_RX_BYTES = 1,
} nvmlPcieUtilCounter_t;

static nvmlReturn_t (*nvmlDeviceGetPcieThroughput)(nvmlDevice_t device, nvmlPcieUtilCounter_t counter,
                                                   unsigned int *value);

static nvmlReturn_t (*nvmlDeviceGetFanSpeed)(nvmlDevice_t device, unsigned int *speed);

typedef enum {
  NVML_TEMPERATURE_GPU = 0,
} nvmlTemperatureSensors_t;

static nvmlReturn_t (*nvmlDeviceGetTemperature)(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType,
                                                unsigned int *temp);

static nvmlReturn_t (*nvmlDeviceGetPowerUsage)(nvmlDevice_t device, unsigned int *power);

static nvmlReturn_t (*nvmlDeviceGetEnforcedPowerLimit)(nvmlDevice_t device, unsigned int *limit);

static nvmlReturn_t (*nvmlDeviceGetEncoderUtilization)(nvmlDevice_t device, unsigned int *utilization,
                                                       unsigned int *samplingPeriodUs);

static nvmlReturn_t (*nvmlDeviceGetDecoderUtilization)(nvmlDevice_t device, unsigned int *utilization,
                                                       unsigned int *samplingPeriodUs);

// Processes running on GPU

typedef struct {
  unsigned int pid;
  unsigned long long usedGpuMemory;
} nvmlProcessInfo_v1_t;

typedef struct {
  unsigned int pid;
  unsigned long long usedGpuMemory;
  unsigned int gpuInstanceId;
  unsigned int computeInstanceId;
} nvmlProcessInfo_v2_t;

typedef struct {
  unsigned int pid;
  unsigned long long usedGpuMemory;
  unsigned int gpuInstanceId;
  unsigned int computeInstanceId;
} nvmlProcessInfo_v3_t;

static nvmlReturn_t (*nvmlDeviceGetComputeRunningProcesses_v1)(nvmlDevice_t device, unsigned int *infoCount,
                                                               nvmlProcessInfo_v1_t *infos);
static nvmlReturn_t (*nvmlDeviceGetComputeRunningProcesses_v2)(nvmlDevice_t device, unsigned int *infoCount,
                                                               nvmlProcessInfo_v2_t *infos);
static nvmlReturn_t (*nvmlDeviceGetComputeRunningProcesses_v3)(nvmlDevice_t device, unsigned int *infoCount,
                                                               nvmlProcessInfo_v3_t *infos);

static nvmlReturn_t (*nvmlDeviceGetComputeRunningProcesses[4])(nvmlDevice_t device, unsigned int *infoCount,
                                                               void *infos);
static void *libixml_handle;

static nvmlReturn_t last_nvml_return_status = NVML_SUCCESS;
static char didnt_call_gpuinfo_init[] = "The ILUVATAR extraction has not been initialized, please call "
                                        "gpuinfo_iluvatar_init\n";
static const char *local_error_string = didnt_call_gpuinfo_init;

// Processes GPU Utilization

struct gpu_info_iluvatar {
  struct gpu_info base;
  struct list_head allocate_list;

  nvmlDevice_t gpuhandle;
  bool isInMigMode;
  unsigned long long last_utilization_timestamp;
};

static LIST_HEAD(allocations);

static bool gpuinfo_iluvatar_init(void);
static void gpuinfo_iluvatar_shutdown(void);
static const char *gpuinfo_iluvatar_last_error_string(void);
static bool gpuinfo_iluvatar_get_device_handles(struct list_head *devices, unsigned *count);
static void gpuinfo_iluvatar_populate_static_info(struct gpu_info *_gpu_info);
static void gpuinfo_iluvatar_refresh_dynamic_info(struct gpu_info *_gpu_info);
static void gpuinfo_iluvatar_get_running_processes(struct gpu_info *_gpu_info);

struct gpu_vendor gpu_vendor_iluvatar = {
    .init = gpuinfo_iluvatar_init,
    .shutdown = gpuinfo_iluvatar_shutdown,
    .last_error_string = gpuinfo_iluvatar_last_error_string,
    .get_device_handles = gpuinfo_iluvatar_get_device_handles,
    .populate_static_info = gpuinfo_iluvatar_populate_static_info,
    .refresh_dynamic_info = gpuinfo_iluvatar_refresh_dynamic_info,
    .refresh_running_processes = gpuinfo_iluvatar_get_running_processes,
    .name = "ILUVATAR",
};

__attribute__((constructor)) static void init_extract_gpuinfo_iluvatar(void) { register_gpu_vendor(&gpu_vendor_iluvatar); }

/*
 *
 * This function loads the libixml.so shared object, initializes the
 * required function pointers and calls the iluvatar library initialization
 * function. Returns true if everything has been initialized successfully. If
 * false is returned, the cause of the error can be retrieved by calling the
 * function gpuinfo_iluvatar_last_error_string.
 *
 */
static bool gpuinfo_iluvatar_init(void) {

  libixml_handle = dlopen("libixml.so", RTLD_LAZY);
  if (!libixml_handle) {
    local_error_string = dlerror();
    return false;
  }

  // Default to last version
  nvmlInit = dlsym(libixml_handle, "nvmlInit_v2");
  if (!nvmlInit)
    nvmlInit = dlsym(libixml_handle, "nvmlInit");
  if (!nvmlInit)
    goto init_error_clean_exit;

  nvmlShutdown = dlsym(libixml_handle, "nvmlShutdown");
  if (!nvmlShutdown)
    goto init_error_clean_exit;

  // Default to last version if available
  nvmlDeviceGetCount = dlsym(libixml_handle, "nvmlDeviceGetCount_v2");
  if (!nvmlDeviceGetCount)
    nvmlDeviceGetCount = dlsym(libixml_handle, "nvmlDeviceGetCount");
  if (!nvmlDeviceGetCount)
    goto init_error_clean_exit;

  nvmlDeviceGetHandleByIndex = dlsym(libixml_handle, "nvmlDeviceGetHandleByIndex_v2");
  if (!nvmlDeviceGetHandleByIndex)
    nvmlDeviceGetHandleByIndex = dlsym(libixml_handle, "nvmlDeviceGetHandleByIndex");
  if (!nvmlDeviceGetHandleByIndex)
    goto init_error_clean_exit;

  nvmlErrorString = dlsym(libixml_handle, "nvmlErrorString");
  if (!nvmlErrorString)
    goto init_error_clean_exit;

  nvmlDeviceGetName = dlsym(libixml_handle, "nvmlDeviceGetName");
  if (!nvmlDeviceGetName)
    goto init_error_clean_exit;

  nvmlDeviceGetPciInfo = dlsym(libixml_handle, "nvmlDeviceGetPciInfo_v3");
  if (!nvmlDeviceGetPciInfo)
    nvmlDeviceGetPciInfo = dlsym(libixml_handle, "nvmlDeviceGetPciInfo_v2");
  if (!nvmlDeviceGetPciInfo)
    nvmlDeviceGetPciInfo = dlsym(libixml_handle, "nvmlDeviceGetPciInfo");
  if (!nvmlDeviceGetPciInfo)
    goto init_error_clean_exit;

  nvmlDeviceGetMaxPcieLinkGeneration = dlsym(libixml_handle, "nvmlDeviceGetMaxPcieLinkGeneration");
  if (!nvmlDeviceGetMaxPcieLinkGeneration)
    goto init_error_clean_exit;

  nvmlDeviceGetMaxPcieLinkWidth = dlsym(libixml_handle, "nvmlDeviceGetMaxPcieLinkWidth");
  if (!nvmlDeviceGetMaxPcieLinkWidth)
    goto init_error_clean_exit;

  nvmlDeviceGetTemperatureThreshold = dlsym(libixml_handle, "nvmlDeviceGetTemperatureThreshold");
  if (!nvmlDeviceGetTemperatureThreshold)
    goto init_error_clean_exit;

  nvmlDeviceGetClockInfo = dlsym(libixml_handle, "nvmlDeviceGetClockInfo");
  if (!nvmlDeviceGetClockInfo)
    goto init_error_clean_exit;

  nvmlDeviceGetMaxClockInfo = dlsym(libixml_handle, "nvmlDeviceGetMaxClockInfo");
  if (!nvmlDeviceGetMaxClockInfo)
    goto init_error_clean_exit;

  nvmlDeviceGetUtilizationRates = dlsym(libixml_handle, "nvmlDeviceGetUtilizationRates");
  if (!nvmlDeviceGetUtilizationRates)
    goto init_error_clean_exit;

  // Get v2 and fallback to v1
  nvmlDeviceGetMemoryInfo_v2 = dlsym(libixml_handle, "nvmlDeviceGetMemoryInfo_v2");
  nvmlDeviceGetMemoryInfo = dlsym(libixml_handle, "nvmlDeviceGetMemoryInfo");
  if (!nvmlDeviceGetMemoryInfo_v2 && !nvmlDeviceGetMemoryInfo)
    goto init_error_clean_exit;

  nvmlDeviceGetCurrPcieLinkGeneration = dlsym(libixml_handle, "nvmlDeviceGetCurrPcieLinkGeneration");
  if (!nvmlDeviceGetCurrPcieLinkGeneration)
    goto init_error_clean_exit;

  nvmlDeviceGetCurrPcieLinkWidth = dlsym(libixml_handle, "nvmlDeviceGetCurrPcieLinkWidth");
  if (!nvmlDeviceGetCurrPcieLinkWidth)
    goto init_error_clean_exit;

  nvmlDeviceGetPcieThroughput = dlsym(libixml_handle, "nvmlDeviceGetPcieThroughput");
  if (!nvmlDeviceGetPcieThroughput)
    goto init_error_clean_exit;

  nvmlDeviceGetFanSpeed = dlsym(libixml_handle, "nvmlDeviceGetFanSpeed");
  if (!nvmlDeviceGetFanSpeed)
    goto init_error_clean_exit;

  nvmlDeviceGetTemperature = dlsym(libixml_handle, "nvmlDeviceGetTemperature");
  if (!nvmlDeviceGetTemperature)
    goto init_error_clean_exit;

  nvmlDeviceGetPowerUsage = dlsym(libixml_handle, "nvmlDeviceGetPowerUsage");
  if (!nvmlDeviceGetPowerUsage)
    goto init_error_clean_exit;

  nvmlDeviceGetEnforcedPowerLimit = dlsym(libixml_handle, "nvmlDeviceGetEnforcedPowerLimit");
  if (!nvmlDeviceGetEnforcedPowerLimit)
    goto init_error_clean_exit;

  nvmlDeviceGetEncoderUtilization = dlsym(libixml_handle, "nvmlDeviceGetEncoderUtilization");
  if (!nvmlDeviceGetEncoderUtilization)
    goto init_error_clean_exit;

  nvmlDeviceGetDecoderUtilization = dlsym(libixml_handle, "nvmlDeviceGetDecoderUtilization");
  if (!nvmlDeviceGetDecoderUtilization)
    goto init_error_clean_exit;

  nvmlDeviceGetComputeRunningProcesses_v3 = dlsym(libixml_handle, "nvmlDeviceGetComputeRunningProcesses_v3");
  nvmlDeviceGetComputeRunningProcesses_v2 = dlsym(libixml_handle, "nvmlDeviceGetComputeRunningProcesses_v2");
  nvmlDeviceGetComputeRunningProcesses_v1 = dlsym(libixml_handle, "nvmlDeviceGetComputeRunningProcesses");
  if (!nvmlDeviceGetComputeRunningProcesses_v3 && !nvmlDeviceGetComputeRunningProcesses_v2 &&
      !nvmlDeviceGetComputeRunningProcesses_v1)
    goto init_error_clean_exit;

  nvmlDeviceGetComputeRunningProcesses[1] =
      (nvmlReturn_t(*)(nvmlDevice_t, unsigned int *, void *))nvmlDeviceGetComputeRunningProcesses_v1;
  nvmlDeviceGetComputeRunningProcesses[2] =
      (nvmlReturn_t(*)(nvmlDevice_t, unsigned int *, void *))nvmlDeviceGetComputeRunningProcesses_v2;
  nvmlDeviceGetComputeRunningProcesses[3] =
      (nvmlReturn_t(*)(nvmlDevice_t, unsigned int *, void *))nvmlDeviceGetComputeRunningProcesses_v3;

  last_nvml_return_status = nvmlInit();
  if (last_nvml_return_status != NVML_SUCCESS) {
    return false;
  }
  local_error_string = NULL;

  return true;

init_error_clean_exit:
  dlclose(libixml_handle);
  libixml_handle = NULL;
  return false;
}

static void gpuinfo_iluvatar_shutdown(void) {
  if (libixml_handle) {
    nvmlShutdown();
    dlclose(libixml_handle);
    libixml_handle = NULL;
    local_error_string = didnt_call_gpuinfo_init;
  }

  struct gpu_info_iluvatar *allocated, *tmp;

  list_for_each_entry_safe(allocated, tmp, &allocations, allocate_list) {
    list_del(&allocated->allocate_list);
    free(allocated);
  }
}

static const char *gpuinfo_iluvatar_last_error_string(void) {
  if (local_error_string) {
    return local_error_string;
  } else if (libixml_handle && nvmlErrorString) {
    return nvmlErrorString(last_nvml_return_status);
  } else {
    return "An unanticipated error occurred while accessing ILUVATAR GPU "
           "information\n";
  }
}

static bool gpuinfo_iluvatar_get_device_handles(struct list_head *devices, unsigned *count) {

  if (!libixml_handle)
    return false;

  unsigned num_devices;
  last_nvml_return_status = nvmlDeviceGetCount(&num_devices);
  if (last_nvml_return_status != NVML_SUCCESS)
    return false;

  struct gpu_info_iluvatar *gpu_infos = calloc(num_devices, sizeof(*gpu_infos));
  if (!gpu_infos) {
    local_error_string = strerror(errno);
    return false;
  }

  list_add(&gpu_infos[0].allocate_list, &allocations);

  *count = 0;
  for (unsigned int i = 0; i < num_devices; ++i) {
    last_nvml_return_status = nvmlDeviceGetHandleByIndex(i, &gpu_infos[*count].gpuhandle);
    if (last_nvml_return_status == NVML_SUCCESS) {
      gpu_infos[*count].base.vendor = &gpu_vendor_iluvatar;
      nvmlPciInfo_t pciInfo;
      nvmlReturn_t pciInfoRet = nvmlDeviceGetPciInfo(gpu_infos[*count].gpuhandle, &pciInfo);
      if (pciInfoRet == NVML_SUCCESS) {
        strncpy(gpu_infos[*count].base.pdev, pciInfo.busIdLegacy, PDEV_LEN);
        list_add_tail(&gpu_infos[*count].base.list, devices);
        *count += 1;
      }
    }
  }

  return true;
}

static void gpuinfo_iluvatar_populate_static_info(struct gpu_info *_gpu_info) {
  struct gpu_info_iluvatar *gpu_info = container_of(_gpu_info, struct gpu_info_iluvatar, base);
  struct gpuinfo_static_info *static_info = &gpu_info->base.static_info;
  nvmlDevice_t device = gpu_info->gpuhandle;

  static_info->integrated_graphics = false;
  static_info->encode_decode_shared = false;
  RESET_ALL(static_info->valid);

  last_nvml_return_status = nvmlDeviceGetName(device, static_info->device_name, MAX_DEVICE_NAME);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_device_name_valid, static_info->valid);

  last_nvml_return_status = nvmlDeviceGetMaxPcieLinkGeneration(device, &static_info->max_pcie_gen);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_max_pcie_gen_valid, static_info->valid);

  last_nvml_return_status = nvmlDeviceGetMaxPcieLinkWidth(device, &static_info->max_pcie_link_width);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_max_pcie_link_width_valid, static_info->valid);

  last_nvml_return_status = nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN,
                                                              &static_info->temperature_shutdown_threshold);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_temperature_shutdown_threshold_valid, static_info->valid);

  last_nvml_return_status = nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN,
                                                              &static_info->temperature_slowdown_threshold);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_temperature_slowdown_threshold_valid, static_info->valid);
}

static void gpuinfo_iluvatar_refresh_dynamic_info(struct gpu_info *_gpu_info) {
  struct gpu_info_iluvatar *gpu_info = container_of(_gpu_info, struct gpu_info_iluvatar, base);
  struct gpuinfo_dynamic_info *dynamic_info = &gpu_info->base.dynamic_info;
  nvmlDevice_t device = gpu_info->gpuhandle;

  bool graphics_clock_valid = false;
  unsigned graphics_clock;
  bool sm_clock_valid = false;
  unsigned sm_clock;
  nvmlClockType_t getMaxClockFrom = NVML_CLOCK_GRAPHICS;

  RESET_ALL(dynamic_info->valid);

  // GPU current speed
  // Maximum between SM and Graphical
  last_nvml_return_status = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &graphics_clock);
  graphics_clock_valid = last_nvml_return_status == NVML_SUCCESS;

  last_nvml_return_status = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_clock);
  sm_clock_valid = last_nvml_return_status == NVML_SUCCESS;

  if (graphics_clock_valid && sm_clock_valid && graphics_clock < sm_clock) {
    getMaxClockFrom = NVML_CLOCK_SM;
  } else if (!graphics_clock_valid && sm_clock_valid) {
    getMaxClockFrom = NVML_CLOCK_SM;
  }

  if (getMaxClockFrom == NVML_CLOCK_GRAPHICS && graphics_clock_valid) {
    SET_GPUINFO_DYNAMIC(dynamic_info, gpu_clock_speed, graphics_clock);
  }
  if (getMaxClockFrom == NVML_CLOCK_SM && sm_clock_valid) {
    SET_GPUINFO_DYNAMIC(dynamic_info, gpu_clock_speed, sm_clock);
  }

  // GPU max speed
  last_nvml_return_status = nvmlDeviceGetMaxClockInfo(device, getMaxClockFrom, &dynamic_info->gpu_clock_speed_max);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_gpu_clock_speed_max_valid, dynamic_info->valid);

  // Memory current speed
  last_nvml_return_status = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &dynamic_info->mem_clock_speed);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_mem_clock_speed_valid, dynamic_info->valid);

  // Memory max speed
  last_nvml_return_status = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_MEM, &dynamic_info->mem_clock_speed_max);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_mem_clock_speed_max_valid, dynamic_info->valid);

  // CPU and Memory utilization rates
  nvmlUtilization_t utilization_percentages;
  last_nvml_return_status = nvmlDeviceGetUtilizationRates(device, &utilization_percentages);
  if (last_nvml_return_status == NVML_SUCCESS) {
    SET_GPUINFO_DYNAMIC(dynamic_info, gpu_util_rate, utilization_percentages.gpu);
  }

  // Encoder utilization rate
  unsigned ignored_period;
  last_nvml_return_status = nvmlDeviceGetEncoderUtilization(device, &dynamic_info->encoder_rate, &ignored_period);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_encoder_rate_valid, dynamic_info->valid);

  // Decoder utilization rate
  last_nvml_return_status = nvmlDeviceGetDecoderUtilization(device, &dynamic_info->decoder_rate, &ignored_period);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_decoder_rate_valid, dynamic_info->valid);

  // Device memory info (total,used,free)
  bool got_meminfo = false;
  if (nvmlDeviceGetMemoryInfo_v2) {
    nvmlMemory_v2_t memory_info;
    memory_info.version = 2;
    last_nvml_return_status = nvmlDeviceGetMemoryInfo_v2(device, &memory_info);
    if (last_nvml_return_status == NVML_SUCCESS) {
      got_meminfo = true;
      SET_GPUINFO_DYNAMIC(dynamic_info, total_memory, memory_info.total);
      SET_GPUINFO_DYNAMIC(dynamic_info, used_memory, memory_info.used);
      SET_GPUINFO_DYNAMIC(dynamic_info, free_memory, memory_info.free);
      SET_GPUINFO_DYNAMIC(dynamic_info, mem_util_rate, memory_info.used * 100 / memory_info.total);
    }
  }
  if (!got_meminfo && nvmlDeviceGetMemoryInfo) {
    nvmlMemory_v1_t memory_info;
    last_nvml_return_status = nvmlDeviceGetMemoryInfo(device, &memory_info);
    if (last_nvml_return_status == NVML_SUCCESS) {
      SET_GPUINFO_DYNAMIC(dynamic_info, total_memory, memory_info.total);
      SET_GPUINFO_DYNAMIC(dynamic_info, used_memory, memory_info.used);
      SET_GPUINFO_DYNAMIC(dynamic_info, free_memory, memory_info.free);
      SET_GPUINFO_DYNAMIC(dynamic_info, mem_util_rate, memory_info.used * 100 / memory_info.total);
    }
  }

  // Pcie generation used by the device
  last_nvml_return_status = nvmlDeviceGetCurrPcieLinkGeneration(device, &dynamic_info->pcie_link_gen);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_pcie_link_gen_valid, dynamic_info->valid);

  // Pcie width used by the device
  last_nvml_return_status = nvmlDeviceGetCurrPcieLinkWidth(device, &dynamic_info->pcie_link_width);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_pcie_link_width_valid, dynamic_info->valid);

  // Pcie reception throughput
  last_nvml_return_status = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_RX_BYTES, &dynamic_info->pcie_rx);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_pcie_rx_valid, dynamic_info->valid);

  // Pcie transmission throughput
  last_nvml_return_status = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_TX_BYTES, &dynamic_info->pcie_tx);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_pcie_tx_valid, dynamic_info->valid);

  // Fan speed
  last_nvml_return_status = nvmlDeviceGetFanSpeed(device, &dynamic_info->fan_speed);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_fan_speed_valid, dynamic_info->valid);

  // GPU temperature
  last_nvml_return_status = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &dynamic_info->gpu_temp);
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_gpu_temp_valid, dynamic_info->valid);

  // Device power usage
  last_nvml_return_status = nvmlDeviceGetPowerUsage(device, &dynamic_info->power_draw);
  dynamic_info->power_draw = dynamic_info->power_draw * 1000; // watts --> milliwatts
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_power_draw_valid, dynamic_info->valid);

  // Maximum enforced power usage
  last_nvml_return_status = nvmlDeviceGetEnforcedPowerLimit(device, &dynamic_info->power_draw_max);
  dynamic_info->power_draw_max = dynamic_info->power_draw_max * 1000; // watts --> milliwatts
  if (last_nvml_return_status == NVML_SUCCESS)
    SET_VALID(gpuinfo_power_draw_max_valid, dynamic_info->valid);

}

static void gpuinfo_iluvatar_get_running_processes(struct gpu_info *_gpu_info) {
  struct gpu_info_iluvatar *gpu_info = container_of(_gpu_info, struct gpu_info_iluvatar, base);
  nvmlDevice_t device = gpu_info->gpuhandle;
  bool validProcessGathering = false;
  for (unsigned version = 3; !validProcessGathering && version > 0; version--) {
    // Get the size of the actual function being used
    size_t sizeof_nvmlProcessInfo;
    switch (version) {
    case 3:
      sizeof_nvmlProcessInfo = sizeof(nvmlProcessInfo_v3_t);
      break;
    case 2:
      sizeof_nvmlProcessInfo = sizeof(nvmlProcessInfo_v2_t);
      break;
    default:
      sizeof_nvmlProcessInfo = sizeof(nvmlProcessInfo_v1_t);
      break;
    }

    _gpu_info->processes_count = 0;
    static size_t array_size = 0;
    static char *retrieved_infos = NULL;
    unsigned int compute_count = 0;
    unsigned int recovered_count = 0;


    if (nvmlDeviceGetComputeRunningProcesses[version]) {
    retry_query_compute:
      recovered_count = array_size;
      last_nvml_return_status = nvmlDeviceGetComputeRunningProcesses[version](
          device, &recovered_count, retrieved_infos);
      if (last_nvml_return_status == NVML_ERROR_INSUFFICIENT_SIZE) {
        array_size += COMMON_PROCESS_LINEAR_REALLOC_INC;
        retrieved_infos = reallocarray(retrieved_infos, array_size, sizeof_nvmlProcessInfo);
        if (!retrieved_infos) {
          perror("Could not re-allocate memory: ");
          exit(EXIT_FAILURE);
        }
        goto retry_query_compute;
      }
      if (last_nvml_return_status == NVML_SUCCESS) {
        validProcessGathering = true;
        compute_count = recovered_count;
      }
    }


    if (!validProcessGathering)
      continue;

    _gpu_info->processes_count = compute_count;
    if (_gpu_info->processes_count > 0) {
      if (_gpu_info->processes_count > _gpu_info->processes_array_size) {
        _gpu_info->processes_array_size = _gpu_info->processes_count + COMMON_PROCESS_LINEAR_REALLOC_INC;
        _gpu_info->processes =
            reallocarray(_gpu_info->processes, _gpu_info->processes_array_size, sizeof(*_gpu_info->processes));
        if (!_gpu_info->processes) {
          perror("Could not allocate memory: ");
          exit(EXIT_FAILURE);
        }
      }
      memset(_gpu_info->processes, 0, _gpu_info->processes_count * sizeof(*_gpu_info->processes));
      for (unsigned i = 0; i < compute_count; ++i) {
        _gpu_info->processes[i].type = gpu_process_compute;
        switch (version) {
        case 2: {
          nvmlProcessInfo_v2_t *pinfo = (nvmlProcessInfo_v2_t *)retrieved_infos;
          _gpu_info->processes[i].pid = pinfo[i].pid;
          _gpu_info->processes[i].gpu_memory_usage = pinfo[i].usedGpuMemory;
        } break;
        case 3: {
          nvmlProcessInfo_v3_t *pinfo = (nvmlProcessInfo_v3_t *)retrieved_infos;
          _gpu_info->processes[i].pid = pinfo[i].pid;
          _gpu_info->processes[i].gpu_memory_usage = pinfo[i].usedGpuMemory;
        } break;
        default: {
          nvmlProcessInfo_v1_t *pinfo = (nvmlProcessInfo_v1_t *)retrieved_infos;
          _gpu_info->processes[i].pid = pinfo[i].pid;
          _gpu_info->processes[i].gpu_memory_usage = pinfo[i].usedGpuMemory;
        } break;
        }
        SET_VALID(gpuinfo_process_gpu_memory_usage_valid, _gpu_info->processes[i].valid);
      }
    }
  }
}
