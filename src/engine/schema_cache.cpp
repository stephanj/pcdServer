#include "pcd/schema_cache.hpp"

namespace pcd {

namespace {
constexpr int failures_before_disable = 2;

std::size_t checkpoint_bytes(const SchemaCacheEntry & entry) {
    return entry.checkpoint ? entry.checkpoint->bytes.size() : 0;
}
}  // namespace

SchemaCache::SchemaCache(SchemaCacheOptions options) : options_(options) {}

std::shared_ptr<const SchemaCacheEntry> SchemaCache::get(const std::string & key) {
    auto it = index_.find(key);
    if (it == index_.end()) {
        return nullptr;
    }
    recency_.splice(recency_.begin(), recency_, it->second.recency);
    return it->second.entry;
}

void SchemaCache::put(const std::string & key, SchemaCacheEntry entry) {
    const auto bytes = checkpoint_bytes(entry);
    if (disabled(key) || bytes > options_.max_bytes || options_.max_entries == 0) {
        return;
    }
    if (auto it = index_.find(key); it != index_.end()) {
        remove_node(it);
    }
    evict_to_fit(bytes);
    recency_.push_front(key);
    index_.emplace(key, Node{std::make_shared<const SchemaCacheEntry>(std::move(entry)), bytes, recency_.begin()});
    total_bytes_ += bytes;
}

void SchemaCache::erase(const std::string & key) {
    if (auto it = index_.find(key); it != index_.end()) {
        remove_node(it);
    }
}

void SchemaCache::record_restore_failure(const std::string & key) {
    erase(key);
    ++restore_failures_[key];
}

bool SchemaCache::disabled(const std::string & key) const {
    auto it = restore_failures_.find(key);
    return it != restore_failures_.end() && it->second >= failures_before_disable;
}

void SchemaCache::clear() {
    recency_.clear();
    index_.clear();
    restore_failures_.clear();
    total_bytes_ = 0;
}

void SchemaCache::evict_to_fit(std::size_t incoming_bytes) {
    while (!recency_.empty() && (index_.size() + 1 > options_.max_entries || total_bytes_ + incoming_bytes > options_.max_bytes)) {
        remove_node(index_.find(recency_.back()));
    }
}

void SchemaCache::remove_node(std::unordered_map<std::string, Node>::iterator it) {
    total_bytes_ -= it->second.bytes;
    recency_.erase(it->second.recency);
    index_.erase(it);
}

}  // namespace pcd
