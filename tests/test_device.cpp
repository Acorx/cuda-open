/**
 * @file test_device.cpp
 * @brief Unit tests for device management
 */

#include <cuda_open/device.h>
#include <iostream>
#include <vector>
#include <cassert>

using namespace cuda_open;

void test_device_initialization() {
    std::cout << "Testing device initialization... ";
    
    DeviceManager::instance().initialize();
    
    auto devices = DeviceManager::instance().get_available_devices();
    assert(!devices.empty());
    
    // Should at least have CPU
    assert(devices[0].type == DeviceType::CPU);
    
    std::cout << "PASSED (found " << devices.size() << " device(s))" << std::endl;
}

void test_device_info() {
    std::cout << "Testing device info... ";
    
    DeviceManager::instance().initialize();
    auto& device = DeviceManager::instance().get_current_device();
    
    auto info = device.get_info();
    assert(!info.name.empty());
    assert(info.num_cores > 0);
    assert(info.type == DeviceType::CPU);
    
    std::cout << "PASSED (" << info.name << ", " << info.num_cores << " cores)" << std::endl;
}

void test_device_selection() {
    std::cout << "Testing device selection... ";
    
    DeviceManager::instance().initialize();
    
    // Should be able to select device 0 (CPU)
    DeviceManager::instance().set_device(0);
    auto& device = DeviceManager::instance().get_current_device();
    assert(device.get_type() == DeviceType::CPU);
    
    std::cout << "PASSED" << std::endl;
}

void test_invalid_device_selection() {
    std::cout << "Testing invalid device selection... ";
    
    DeviceManager::instance().initialize();
    
    bool caught = false;
    try {
        DeviceManager::instance().set_device(999);
    } catch (const DeviceError&) {
        caught = true;
    }
    assert(caught);
    
    std::cout << "PASSED" << std::endl;
}

void test_memory_operations() {
    std::cout << "Testing memory operations... ";
    
    DeviceManager::instance().initialize();
    auto& device = DeviceManager::instance().get_current_device();
    
    // Allocate
    const size_t size = 1024;
    void* ptr = device.allocate(size);
    assert(ptr != nullptr);
    
    // Write and read
    std::vector<float> data(size / sizeof(float), 3.14f);
    device.memcpy_host_to_device(ptr, data.data(), size);
    
    std::vector<float> recv(size / sizeof(float));
    device.memcpy_device_to_host(recv.data(), ptr, size);
    
    // Verify
    for (auto val : recv) {
        assert(val == 3.14f);
    }
    
    // Free
    device.free(ptr);
    
    std::cout << "PASSED" << std::endl;
}

void test_synchronization() {
    std::cout << "Testing synchronization... ";
    
    DeviceManager::instance().initialize();
    auto& device = DeviceManager::instance().get_current_device();
    
    // Should not throw
    device.synchronize();
    
    std::cout << "PASSED" << std::endl;
}

void test_stream_operations() {
    std::cout << "Testing stream operations... ";
    
    DeviceManager::instance().initialize();
    auto& device = DeviceManager::instance().get_current_device();
    
    // Create and destroy stream
    void* stream = device.create_stream();
    device.destroy_stream(stream);
    
    std::cout << "PASSED" << std::endl;
}

void test_gpu_availability_check() {
    std::cout << "Testing GPU availability check... ";
    
    DeviceManager::instance().initialize();
    bool has_gpu = DeviceManager::instance().is_gpu_available();
    
    // Result depends on system configuration
    std::cout << "PASSED (GPU " << (has_gpu ? "available" : "not available") << ")" << std::endl;
}

int main() {
    std::cout << "=== Running Device Tests ===\n" << std::endl;
    
    try {
        test_device_initialization();
        test_device_info();
        test_device_selection();
        test_invalid_device_selection();
        test_memory_operations();
        test_synchronization();
        test_stream_operations();
        test_gpu_availability_check();
        
        std::cout << "\nAll device tests PASSED!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED: " << e.what() << std::endl;
        return 1;
    }
}
