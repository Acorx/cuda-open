/**
 * @example 01_basic_memory.cpp
 * @brief Demonstrates basic memory allocation and transfers
 */

#include <cuda_open/cuda_open.h>
#include <iostream>
#include <vector>

int main() {
    try {
        // Initialize library
        cuda_open::initialize();
        
        std::cout << "=== CUDA Open - Basic Memory Example ===\n" << std::endl;
        std::cout << cuda_open::get_info() << std::endl;
        
        // Get current device
        auto& device = cuda_open::DeviceManager::instance().get_current_device();
        std::cout << "Using device: " << device.get_name() << std::endl;
        
        // Allocate device memory
        const size_t size = 1024;
        cuda_open::DevicePtr<float> dev_data(device, size);
        std::cout << "Allocated " << size << " floats on device" << std::endl;
        
        // Prepare host data
        std::vector<float> host_data(size);
        for (size_t i = 0; i < size; ++i) {
            host_data[i] = static_cast<float>(i) * 0.001f;
        }
        
        // Copy to device
        dev_data.copy_from_host(host_data.data());
        std::cout << "Copied data from host to device" << std::endl;
        
        // Copy back from device
        std::vector<float> host_data_recv(size);
        dev_data.copy_to_host(host_data_recv.data());
        std::cout << "Copied data from device to host" << std::endl;
        
        // Verify
        bool match = true;
        for (size_t i = 0; i < size; ++i) {
            if (std::abs(host_data[i] - host_data_recv[i]) > 1e-6) {
                match = false;
                break;
            }
        }
        std::cout << "Data verification: " << (match ? "PASSED" : "FAILED") << std::endl;
        
        // Unified buffer example
        cuda_open::UnifiedBuffer<double> buffer(device, 512);
        std::cout << "\nCreated unified buffer with " << buffer.size() << " doubles" << std::endl;
        
        // Initialize host side
        for (size_t i = 0; i < buffer.size(); ++i) {
            buffer.host_data()[i] = static_cast<double>(i) * 2.0;
        }
        
        // Sync to device
        buffer.sync_to_device();
        std::cout << "Synced unified buffer to device" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
