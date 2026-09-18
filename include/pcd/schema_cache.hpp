#pragma once
#include "pcd/llama_runtime.hpp"
#include "pcd/schema.hpp"

#include <cstddef>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>

namespace pcd {

struct SchemaCacheOptions {
    std::size_t max_entries{32};
    std::size_t max_bytes{512ull * 1024 * 1024};
};

// Immutable once inserted: readers hold shared_ptr copies that outlive eviction.
struct SchemaCacheEntry {
    std::shared_ptr<const CompiledSchema> schema;
    std::shared_ptr<const SequenceCheckpoint> checkpoint;  // may be null when caching is disabled
};

// Entry- and byte-bounded LRU of compiled schemas and their prefix checkpoints.
// Not thread-safe; the owning engine serializes access.
class SchemaCache {
public:
    explicit SchemaCache(SchemaCacheOptions options = {});

    // Returns the entry and marks it most recently used, or null.
    std::shared_ptr<const SchemaCacheEntry> get(const std::string & key);
    // Inserts or replaces. Ignored when the key is disabled or the checkpoint
    // alone exceeds the byte budget.
    void put(const std::string & key, SchemaCacheEntry entry);
    void erase(const std::string & key);
    // Drops the key's checkpoint; the second failure disables the key.
    void record_restore_failure(const std::string & key);
    bool disabled(const std::string & key) const;
    // Removes every entry and every disablement (model reset).
    void clear();

    std::size_t size() const { return index_.size(); }
    std::size_t total_bytes() const { return total_bytes_; }
    const SchemaCacheOptions & options() const { return options_; }

private:
    struct Node {
        std::shared_ptr<const SchemaCacheEntry> entry;
        std::size_t bytes;
        std::list<std::string>::iterator recency;
    };

    void evict_to_fit(std::size_t incoming_bytes);
    void remove_node(std::unordered_map<std::string, Node>::iterator it);

    SchemaCacheOptions options_;
    std::list<std::string> recency_;  // front = most recently used
    std::unordered_map<std::string, Node> index_;
    std::unordered_map<std::string, int> restore_failures_;
    std::size_t total_bytes_{0};
};

}  // namespace pcd
