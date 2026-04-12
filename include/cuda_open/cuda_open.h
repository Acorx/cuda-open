/**
 * @file cuda_open.h
 * @brief Main include header for CUDA Open library
 * 
 * Single header to include all CUDA Open functionality.
 */

#ifndef CUDA_OPEN_H
#define CUDA_OPEN_H

#include "device.h"
#include "memory.h"
#include "quantization.h"
#include "kernel.h"
#include "tensor.h"

// Version information
#define CUDA_OPEN_VERSION_MAJOR 0
#define CUDA_OPEN_VERSION_MINOR 1
#define CUDA_OPEN_VERSION_PATCH 0

namespace cuda_open {

// Get version string
inline const char* get_version() {
    return "0.1.0";
}

// Initialize the library
inline void initialize() {
    DeviceManager::instance().initialize();
}

// Get library info
inline std::string get_info() {
    std::string info = "CUDA Open v" + std::string(get_version()) + "\n";
    info += "Available devices:\n";
    
    auto devices = DeviceManager::instance().get_available_devices();
    for (size_t i = 0; i < devices.size(); ++i) {
        info += "  [" + std::to_string(i) + "] " + devices[i].name;
        info += devices[i].type == DeviceType::CPU ? " (CPU)" : " (GPU)";
        info += "\n";
    }
    
    return info;
}

} // namespace cuda_open

#endif // CUDA_OPEN_H
