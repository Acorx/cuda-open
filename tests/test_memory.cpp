/**
 * @file test_memory.cpp
 * @brief Unit tests for memory management
 */

#include <cuda_open/memory.h>
#include <cuda_open/device.h>
#include <iostream>
#include <vector>
#include <cassert>

using namespace cuda_open;

void test_device_ptr_allocation() {
    std::cout << "Testing DevicePtr allocation... ";
    
    // Initialize and get CPU device
    DeviceManager::instance().initialize();
    auto& device = DeviceManager::instance().get_current_device();
    
    const size_t size = 1024;
    DevicePtr<float> ptr(device, size);
    
    assert(ptr.valid());
    assert(ptr.size() == size);
    assert(ptr.device() == &device);
    
    std::cout << "PASSED" << std::endl;
}

void test_host_device_copy() {
    std::cout << "Testing host-device copy... ";
    
    DeviceManager::instance().initialize();
    auto& device = DeviceManager::instance().get_current_device();
    
    const size_t size = 512;
    DevicePtr<float> ptr(device, size);
    
    // Prepare host data
    std::vector<float> host_data(size);
    for (size_t i = 0; i < size; ++i) {
        host_data[i] = static_cast<float>(i) * 0.01f;
    }
    
    // Copy to device
    ptr.copy_from_host(host_data.data());
    
    // Copy back
    std::vector<float> host_data_recv(size);
    ptr.copy_to_host(host_data_recv.data());
    
    // Verify
    float max_error = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        max_error = std::max(max_error, std::abs(host_data[i] - host_data_recv[i]));
    }
    assert(max_error < 1e-6f);
    
    std::cout << "PASSED (max_error=" << max_error << ")" << std::endl;
}

void test_unified_buffer() {
    std::cout << "Testing UnifiedBuffer... ";
    
    DeviceManager::instance().initialize();
    auto& device = DeviceManager::instance().get_current_device();
    
    const size_t size = 256;
    UnifiedBuffer<double> buffer(device, size);
    
    assert(buffer.size() == size);
    
    // Initialize host data
    for (size_t i = 0; i < size; ++i) {
        buffer.host_data()[i] = static_cast<double>(i) * 2.5;
    }
    
    // Sync to device
    buffer.sync_to_device();
    
    // Modify host data
    for (size_t i = 0; i < size; ++i) {
        buffer.host_data()[i] = 0.0;
    }
    
    // Sync from device
    buffer.sync_from_device();
    
    // Verify data restored from device
    for (size_t i = 0; i < size; ++i) {
        double expected = static_cast<double>(i) * 2.5;
        assert(std::abs(buffer.host_data()[i] - expected) < 1e-9);
    }
    
    std::cout << "PASSED" << std::endl;
}

void test_move_semantics() {
    std::cout << "Testing move semantics... ";
    
    DeviceManager::instance().initialize();
    auto& device = DeviceManager::instance().get_current_device();
    
    const size_t size = 128;
    DevicePtr<int> ptr1(device, size);
    
    std::vector<int> data(size, 42);
    ptr1.copy_from_host(data.data());
    
    // Move
    DevicePtr<int> ptr2(std::move(ptr1));
    assert(ptr2.valid());
    assert(ptr2.size() == size);
    assert(!ptr1.valid());  // Original should be invalid
    
    // Verify data still accessible
    std::vector<int> recv(size);
    ptr2.copy_to_host(recv.data());
    for (auto val : recv) {
        assert(val == 42);
    }
    
    std::cout << "PASSED" << std::endl;
}

void test_multiple_allocations() {
    std::cout << "Testing multiple allocations... ";
    
    DeviceManager::instance().initialize();
    auto& device = DeviceManager::instance().get_current_device();
    
    std::vector<DevicePtr<float>> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.emplace_back(device, 100);
        assert(ptrs.back().valid());
    }
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "=== Running Memory Tests ===\n" << std::endl;
    
    try {
        test_device_ptr_allocation();
        test_host_device_copy();
        test_unified_buffer();
        test_move_semantics();
        test_multiple_allocations();
        
        std::cout << "\nAll memory tests PASSED!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED: " << e.what() << std::endl;
        return 1;
    }
}
