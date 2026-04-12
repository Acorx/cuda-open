/**
 * @file paged_attention.h
 * @brief PagedAttention Implementation (vLLM-style Memory Management)
 * 
 * Revolutionary KV cache memory management:
 * - Non-contiguous KV cache storage
 * - Page-level allocation (like virtual memory)
 * - Zero-copy attention computation
 * - Memory sharing across sequences
 * 
 * Benefits:
 * - 2-4x memory efficiency vs contiguous cache
 * - Zero fragmentation
 * - Dynamic sequence length support
 * - Batch optimization
 * 
 * Reference: Kwon et al. "vLLM: Easy, Fast, and Cheap LLM Serving" (2023)
 */

#ifndef CUDA_OPEN_PAGED_ATTENTION_H
#define CUDA_OPEN_PAGED_ATTENTION_H

#include <vector>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>

namespace cuda_open {
namespace optimization {

// ============================================================================
// BLOCK TABLE & PAGE MANAGEMENT
// ============================================================================

/**
 * KV Cache page structure
 * Stores KV states for a fixed number of tokens
 */
struct KVBlock {
    int block_id;
    int block_size;  // Number of tokens this block can store
    float* k_cache;  // [block_size, num_heads, head_dim]
    float* v_cache;  // [block_size, num_heads, head_dim]
    int num_tokens_stored;
    bool is_shared;  // true if shared across sequences
    
    KVBlock() 
        : block_id(-1), block_size(0), k_cache(nullptr), 
          v_cache(nullptr), num_tokens_stored(0), is_shared(false) {}
    
    ~KVBlock() {
        delete[] k_cache;
        delete[] v_cache;
    }
};

/**
 * Block table for a sequence
 * Maps logical token positions to physical blocks
 */
struct BlockTable {
    int sequence_id;
    std::vector<int> block_indices;  // Physical block indices
    int logical_length;  // Number of tokens in sequence
    int max_blocks;  // Maximum blocks allocated
    
    int get_num_blocks() const { return block_indices.size(); }
    
    int get_block_index(int token_pos, int block_size) const {
        return block_indices[token_pos / block_size];
    }
    
    int get_offset_in_block(int token_pos, int block_size) const {
        return token_pos % block_size;
    }
};

// ============================================================================
// PAGED KV CACHE MANAGER
// ============================================================================

class PagedKVCacheManager {
public:
    /**
     * Configuration for paged KV cache
     */
    struct Config {
        int block_size = 16;  // Tokens per block (like page size)
        int num_blocks = 1024;  // Total blocks in pool
        int num_heads = 32;
        int head_dim = 128;
        bool enable_sharing = true;  // Allow cross-sequence sharing
    };
    
    PagedKVCacheManager(const Config& config = Config())
        : config_(config) {
        // Allocate block pool
        block_pool_.resize(config.num_blocks);
        free_blocks_.resize(config.num_blocks);
        
        for (int i = 0; i < config.num_blocks; ++i) {
            block_pool_[i] = std::make_unique<KVBlock>();
            block_pool_[i]->block_id = i;
            block_pool_[i]->block_size = config.block_size;
            
            int k_size = config.block_size * config.num_heads * config.head_dim;
            block_pool_[i]->k_cache = new float[k_size];
            block_pool_[i]->v_cache = new float[k_size];
            block_pool_[i]->num_tokens_stored = 0;
            
            free_blocks_[i] = i;
        }
        
        std::cout << "PagedKVCacheManager initialized:" << std::endl;
        std::cout << "  Block size: " << config.block_size << " tokens" << std::endl;
        std::cout << "  Total blocks: " << config.num_blocks << std::endl;
        std::cout << "  Memory: " << (config.num_blocks * config.block_size * 
                        config.num_heads * config.head_dim * 2 * sizeof(float) / 1024 / 1024) 
                  << " MB" << std::endl;
    }
    
    /**
     * Allocate a new block for a sequence
     */
    int allocate_block(int sequence_id) {
        if (free_blocks_.empty()) {
            throw std::runtime_error("No free blocks available (OOM)");
        }
        
        int block_idx = free_blocks_.back();
        free_blocks_.pop_back();
        
        block_pool_[block_idx]->num_tokens_stored = 0;
        block_pool_[block_idx]->is_shared = false;
        
        // Track in sequence map
        if (sequence_block_tables_.find(sequence_id) == sequence_block_tables_.end()) {
            sequence_block_tables_[sequence_id] = BlockTable();
            sequence_block_tables_[sequence_id].sequence_id = sequence_id;
            sequence_block_tables_[sequence_id].logical_length = 0;
        }
        
        sequence_block_tables_[sequence_id].block_indices.push_back(block_idx);
        
        return block_idx;
    }
    
    /**
     * Free a block and return to pool
     */
    void free_block(int sequence_id, int block_idx) {
        auto it = sequence_block_tables_.find(sequence_id);
        if (it != sequence_block_tables_.end()) {
            auto& blocks = it->second.block_indices;
            blocks.erase(std::remove(blocks.begin(), blocks.end(), block_idx), blocks.end());
        }
        
        block_pool_[block_idx]->num_tokens_stored = 0;
        free_blocks_.push_back(block_idx);
    }
    
    /**
     * Free all blocks for a sequence
     */
    void free_sequence(int sequence_id) {
        auto it = sequence_block_tables_.find(sequence_id);
        if (it != sequence_block_tables_.end()) {
            for (int block_idx : it->second.block_indices) {
                free_blocks_.push_back(block_idx);
                block_pool_[block_idx]->num_tokens_stored = 0;
            }
            sequence_block_tables_.erase(it);
        }
    }
    
    /**
     * Append tokens to KV cache
     * Allocates new blocks as needed
     */
    void append_tokens(
        int sequence_id,
        const float* k_states,  // [num_tokens, num_heads, head_dim]
        const float* v_states,  // [num_tokens, num_heads, head_dim]
        int num_tokens
    ) {
        auto& table = sequence_block_tables_[sequence_id];
        
        for (int i = 0; i < num_tokens; ++i) {
            int token_pos = table.logical_length + i;
            int block_idx = token_pos / config_.block_size;
            int offset_in_block = token_pos % config_.block_size;
            
            // Allocate new block if needed
            if (block_idx >= table.get_num_blocks()) {
                allocate_block(sequence_id);
            }
            
            int physical_block = table.block_indices[block_idx];
            auto& block = block_pool_[physical_block];
            
            // Copy K and V states to block
            int k_stride = config_.num_heads * config_.head_dim;
            std::memcpy(
                block->k_cache + offset_in_block * k_stride,
                k_states + i * k_stride,
                k_stride * sizeof(float)
            );
            std::memcpy(
                block->v_cache + offset_in_block * k_stride,
                v_states + i * k_stride,
                k_stride * sizeof(float)
            );
            
            block->num_tokens_stored = std::max(
                block->num_tokens_stored, offset_in_block + 1
            );
        }
        
        table.logical_length += num_tokens;
    }
    
    /**
     * Get KV cache for attention computation
     * Returns flattened view across all blocks
     */
    struct KVView {
        const float* k_data;
        const float* v_data;
        int total_tokens;
        std::vector<std::pair<int, int>> block_offsets;  // (block_idx, num_tokens)
    };
    
    KVView get_kv_view(int sequence_id) const {
        auto it = sequence_block_tables_.find(sequence_id);
        if (it == sequence_block_tables_.end()) {
            return {nullptr, nullptr, 0, {}};
        }
        
        const auto& table = it->second;
        KVView view;
        view.total_tokens = table.logical_length;
        
        // Collect block offsets
        int num_full_blocks = table.logical_length / config_.block_size;
        int remainder = table.logical_length % config_.block_size;
        
        for (int i = 0; i < num_full_blocks; ++i) {
            view.block_offsets.push_back({
                table.block_indices[i], config_.block_size
            });
        }
        
        if (remainder > 0 && num_full_blocks < table.get_num_blocks()) {
            view.block_offsets.push_back({
                table.block_indices[num_full_blocks], remainder
            });
        }
        
        return view;
    }
    
    /**
     * Share KV cache blocks between sequences
     * Used for prefix caching in multi-turn conversations
     */
    void share_prefix_blocks(int src_sequence, int dst_sequence, int prefix_length) {
        int num_blocks_to_share = (prefix_length + config_.block_size - 1) / config_.block_size;
        
        auto& src_table = sequence_block_tables_[src_sequence];
        auto& dst_table = sequence_block_tables_[dst_sequence];
        
        // Share blocks
        for (int i = 0; i < num_blocks_to_share && i < src_table.get_num_blocks(); ++i) {
            int src_block = src_table.block_indices[i];
            block_pool_[src_block]->is_shared = true;
            dst_table.block_indices.push_back(src_block);
        }
        
        dst_table.logical_length = std::min(prefix_length, src_table.logical_length);
    }
    
    /**
     * Get memory usage statistics
     */
    struct MemoryStats {
        int total_blocks;
        int used_blocks;
        int free_blocks;
        float utilization;
        int total_sequences;
    };
    
    MemoryStats get_stats() const {
        MemoryStats stats;
        stats.total_blocks = config_.num_blocks;
        stats.free_blocks = free_blocks_.size();
        stats.used_blocks = stats.total_blocks - stats.free_blocks;
        stats.utilization = (float)stats.used_blocks / stats.total_blocks;
        stats.total_sequences = sequence_block_tables_.size();
        return stats;
    }

private:
    Config config_;
    std::vector<std::unique_ptr<KVBlock>> block_pool_;
    std::vector<int> free_blocks_;
    std::unordered_map<int, BlockTable> sequence_block_tables_;
};

// ============================================================================
// PAGED ATTENTION COMPUTATION
// ============================================================================

class PagedAttentionKernel {
public:
    /**
     * Compute attention with paged KV cache
     * 
     * For each head:
     *   1. Load Q from current position
     *   2. Iterate through KV blocks
     *   3. Compute attention scores
     *   4. Softmax and accumulate
     */
    static void compute_attention(
        const float* query,           // [num_heads, head_dim]
        const PagedKVCacheManager::KVView& kv_view,
        float* output,                // [num_heads, head_dim]
        int num_heads,
        int head_dim,
        const PagedKVCacheManager& cache_manager
    ) {
        int block_size = cache_manager.get_config().block_size;
        int k_stride = num_heads * head_dim;
        
        // Initialize output
        std::memset(output, 0, num_heads * head_dim * sizeof(float));
        
        // Compute attention for each head
        for (int h = 0; h < num_heads; ++h) {
            const float* q_head = query + h * head_dim;
            float* out_head = output + h * head_dim;
            
            // Accumulators for softmax
            float max_score = -1e20f;
            float sum_exp = 0.0f;
            
            // First pass: find max score for numerical stability
            std::vector<float> scores;
            scores.reserve(kv_view.total_tokens);
            
            int token_idx = 0;
            for (const auto& [block_idx, num_tokens] : kv_view.block_offsets) {
                const auto& block = cache_manager.get_block(block_idx);
                
                for (int t = 0; t < num_tokens; ++t) {
                    const float* k_head = block->k_cache + t * k_stride + h * head_dim;
                    
                    // Compute Q·K^T
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        score += q_head[d] * k_head[d];
                    }
                    score /= std::sqrt(head_dim);  // Scaling
                    
                    scores.push_back(score);
                    max_score = std::max(max_score, score);
                    token_idx++;
                }
            }
            
            // Second pass: softmax and accumulate
            token_idx = 0;
            for (const auto& [block_idx, num_tokens] : kv_view.block_offsets) {
                const auto& block = cache_manager.get_block(block_idx);
                
                for (int t = 0; t < num_tokens; ++t) {
                    float score = scores[token_idx];
                    float weight = std::exp(score - max_score);
                    sum_exp += weight;
                    
                    // Accumulate V * weight
                    const float* v_head = block->v_cache + t * k_stride + h * head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        out_head[d] += weight * v_head[d];
                    }
                    
                    token_idx++;
                }
            }
            
            // Normalize
            if (sum_exp > 1e-10f) {
                for (int d = 0; d < head_dim; ++d) {
                    out_head[d] /= sum_exp;
                }
            }
        }
    }
    
    /**
     * Multi-query attention (MQA/GQA optimization)
     * Shares KV cache across query groups
     */
    static void compute_grouped_query_attention(
        const float* query,           // [num_query_heads, head_dim]
        const PagedKVCacheManager::KVView& kv_view,
        float* output,                // [num_query_heads, head_dim]
        int num_query_heads,
        int num_kv_heads,
        int head_dim,
        const PagedKVCacheManager& cache_manager
    ) {
        int heads_per_kv = num_query_heads / num_kv_heads;
        
        for (int kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
            // Get shared KV for this group
            std::vector<const float*> k_heads, v_heads;
            
            int token_idx = 0;
            for (const auto& [block_idx, num_tokens] : kv_view.block_offsets) {
                const auto& block = cache_manager.get_block(block_idx);
                int k_stride = num_kv_heads * head_dim;
                
                for (int t = 0; t < num_tokens; ++t) {
                    k_heads.push_back(block->k_cache + t * k_stride + kv_h * head_dim);
                    v_heads.push_back(block->v_cache + t * k_stride + kv_h * head_dim);
                    token_idx++;
                }
            }
            
            // Compute attention for all query heads in this group
            for (int q = 0; q < heads_per_kv; ++q) {
                int query_h = kv_h * heads_per_kv + q;
                const float* q_head = query + query_h * head_dim;
                float* out_head = output + query_h * head_dim;
                
                // Standard attention computation
                float max_score = -1e20f;
                float sum_exp = 0.0f;
                std::vector<float> scores(k_heads.size());
                
                // First pass
                for (size_t i = 0; i < k_heads.size(); ++i) {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        score += q_head[d] * k_heads[i][d];
                    }
                    score /= std::sqrt(head_dim);
                    scores[i] = score;
                    max_score = std::max(max_score, score);
                }
                
                // Second pass
                std::memset(out_head, 0, head_dim * sizeof(float));
                for (size_t i = 0; i < k_heads.size(); ++i) {
                    float weight = std::exp(scores[i] - max_score);
                    sum_exp += weight;
                    for (int d = 0; d < head_dim; ++d) {
                        out_head[d] += weight * v_heads[i][d];
                    }
                }
                
                if (sum_exp > 1e-10f) {
                    for (int d = 0; d < head_dim; ++d) {
                        out_head[d] /= sum_exp;
                    }
                }
            }
        }
    }
};

// ============================================================================
// SEQUENCE MANAGER WITH PAGED ATTENTION
// ============================================================================

class PagedAttentionSequenceManager {
public:
    PagedAttentionSequenceManager(int max_sequences = 128)
        : max_sequences_(max_sequences), next_sequence_id_(0) {}
    
    /**
     * Add a new sequence
     */
    int add_sequence() {
        int seq_id = next_sequence_id_++;
        sequence_lengths_[seq_id] = 0;
        sequence_is_active_[seq_id] = true;
        return seq_id;
    }
    
    /**
     * Remove a sequence and free its blocks
     */
    void remove_sequence(int seq_id, PagedKVCacheManager& cache_manager) {
        cache_manager.free_sequence(seq_id);
        sequence_lengths_.erase(seq_id);
        sequence_is_active_.erase(seq_id);
    }
    
    /**
     * Append tokens to a sequence
     */
    void append_tokens(
        int seq_id,
        const std::vector<int>& tokens,
        PagedKVCacheManager& cache_manager,
        const float* k_states,
        const float* v_states
    ) {
        cache_manager.append_tokens(
            seq_id, k_states, v_states, tokens.size()
        );
        sequence_lengths_[seq_id] += tokens.size();
    }
    
    /**
     * Get sequences for batching
     */
    std::vector<int> get_active_sequences() const {
        std::vector<int> active;
        for (const auto& [seq_id, is_active] : sequence_is_active_) {
            if (is_active) {
                active.push_back(seq_id);
            }
        }
        return active;
    }

private:
    int max_sequences_;
    int next_sequence_id_;
    std::unordered_map<int, int> sequence_lengths_;
    std::unordered_map<int, bool> sequence_is_active_;
};

} // namespace optimization
} // namespace cuda_open

#endif // CUDA_OPEN_PAGED_ATTENTION_H
