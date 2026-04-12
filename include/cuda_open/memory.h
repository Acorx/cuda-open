/**
 * @file memory.h
 * @brief Unified memory management with automatic CPU/GPU handling
 * 
 * Provides smart memory allocation that works across CPU and GPU
 * with automatic data transfer and synchronization.
 */

#ifndef CUDA_OPEN_MEMORY_H
#define CUDA_OPEN_MEMORY_H

#include "device.h"
#include <cstddef>
#include <memory>
#include <type_traits>

namespace cuda_open {

// Memory pointer wrapper with device tracking
template<typename T>
class DevicePtr {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    
    DevicePtr() : ptr_(nullptr), device_(nullptr), size_(0) {}
    
    explicit DevicePtr(Device& device, size_t count) 
        : ptr_(nullptr), device_(&device), size_(count) {
        ptr_ = static_cast<T*>(device.allocate(count * sizeof(T)));
    }
    
    ~DevicePtr() {
        if (ptr_ && device_) {
            device_->free(ptr_);
        }
    }
    
    // Move semantics
    DevicePtr(DevicePtr&& other) noexcept 
        : ptr_(other.ptr_), device_(other.device_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.device_ = nullptr;
        other.size_ = 0;
    }
    
    DevicePtr& operator=(DevicePtr&& other) noexcept {
        if (this != &other) {
            if (ptr_ && device_) {
                device_->free(ptr_);
            }
            ptr_ = other.ptr_;
            device_ = other.device_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.device_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Copy to/from host
    void copy_from_host(const T* host_data) {
        if (!ptr_ || !device_) throw DeviceError("Invalid device pointer");
        device_->memcpy_host_to_device(ptr_, host_data, size_ * sizeof(T));
    }
    
    void copy_to_host(T* host_data) const {
        if (!ptr_ || !device_) throw DeviceError("Invalid device pointer");
        device_->memcpy_device_to_host(host_data, ptr_, size_ * sizeof(T));
    }
    
    // Accessors
    pointer get() const { return ptr_; }
    Device* device() const { return device_; }
    size_t size() const { return size_; }
    bool valid() const { return ptr_ != nullptr; }
    
    explicit operator bool() const { return valid(); }
    
private:
    T* ptr_;
    Device* device_;
    size_t size_;
};

// Unified memory buffer (automatically manages host and device copies)
template<typename T>
class UnifiedBuffer {
public:
    UnifiedBuffer() : host_data_(nullptr), device_ptr_{}, dirty_(false) {}
    
    UnifiedBuffer(Device& device, size_t count) 
        : host_data_(new T[count]()), device_ptr_(device, count), 
          size_(count), dirty_(false) {}
    
    ~UnifiedBuffer() {
        delete[] host_data_;
    }
    
    // Access host data
    T* host_data() { return host_data_; }
    const T* host_data() const { return host_data_; }
    
    // Access device pointer
    DevicePtr<T>& device_ptr() { return device_ptr_; }
    const DevicePtr<T>& device_ptr() const { return device_ptr_; }
    
    // Sync to device
    void sync_to_device() {
        if (device_ptr_.valid()) {
            device_ptr_.copy_from_host(host_data_);
            dirty_ = false;
        }
    }
    
    // Sync from device
    void sync_from_device() {
        if (device_ptr_.valid()) {
            device_ptr_.copy_to_host(host_data_);
            dirty_ = true;
        }
    }
    
    size_t size() const { return size_; }
    
private:
    T* host_data_;
    DevicePtr<T> device_ptr_;
    size_t size_;
    bool dirty_;
};

} // namespace cuda_open

#endif // CUDA_OPEN_MEMORY_H
