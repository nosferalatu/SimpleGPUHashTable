/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of The Unlicense.​
 * If a copy of the license was not distributed with this file, ​
 * you can obtain one at https://spdx.org/licenses/Unlicense.html​
 *​
 *
 * SPDX-License-Identifier: Unlicense
 */

#include "stdio.h"
#include "stdint.h"
#include "vector"

#define CPP_MODULE "KERNEL"
#include "linearprobing.h"

#include <sycl/sycl.hpp>
#include <chrono>
#include "acas.h"

// 32 bit Murmur3 hash
uint32_t hash(uint32_t k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (kHashTableCapacity - 1);
}

// Create a hash table. For linear probing, this is just an array of KeyValues
KeyValue* create_hashtable(sycl::queue& qht)
{
    KeyValue* hashtable;

    try {
        // Allocate memory
        hashtable = sycl::malloc_device<KeyValue>(kHashTableCapacity, qht);

        // Initialize hash table to empty
        static_assert(kEmpty == 0xFFFFFFFF, "memset expected kEmpty=0xFFFFFFFF");
        qht.memset(hashtable, 0xFF, sizeof(KeyValue) * kHashTableCapacity);
        qht.wait();
    } catch (std::exception const& e) {
        LOG_ERROR("Exception caught, \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception caught, bailing...");
    }

    return hashtable;
}

// Insert the key/values in kvs into the hashtable
void gpu_hashtable_insert(
    KeyValue* hashtable,
    const KeyValue* kvs,
    unsigned int numkvs,
    sycl::nd_item<1> item)
{
    unsigned int tid = item.get_global_id(0);
    if (tid < numkvs) {
        uint32_t key   = kvs[tid].key;
        uint32_t value = kvs[tid].value;
        uint32_t slot  = hash(key);

        while (true) {
            uint32_t prev = acas::atomic_compare_exchange_strong(&hashtable[slot].key, kEmpty, key);
            if (prev == kEmpty || prev == key) {
                hashtable[slot].value = value;
                return;
            }

            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

void insert_hashtable(
    KeyValue* pHashTable, // hashtable
    const KeyValue* kvs,  // starting position for this batch of key-value pairs
    uint32_t num_kvs,     // number of key-value pairs in this batch
    sycl::queue& qht)
{
    try {
        // Copy this batch of key-value pairs to the device
        KeyValue* device_kvs;
        device_kvs = sycl::malloc_device<KeyValue>(num_kvs, qht);
        auto e1 = qht.memcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs);

        int threadblocksize = 256; // perf does not seem to vary w/ thread block size (for all kernels in hashtable)

        // Create events for GPU timing
        qht.parallel_for(
            sycl::nd_range<1>(num_kvs, threadblocksize), std::move(e1),
            [=](sycl::nd_item<1> item) {

                gpu_hashtable_insert(
                    pHashTable,
                    device_kvs,
                    (uint32_t)num_kvs,
                    item);
            }
        );
        qht.wait();

        sycl::free(device_kvs, qht);
    } catch (std::exception const& e) {
        LOG_ERROR("Exception caught, \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception caught, bailing...");
    }
}

// Lookup keys in the hashtable, and return the values
void gpu_hashtable_lookup(
    KeyValue* hashtable,
    KeyValue* kvs,
    unsigned int numkvs,
    sycl::nd_item<1> item)
{
    unsigned int tid = item.get_global_id(0);
    if (tid < numkvs) {
        uint32_t key  = kvs[tid].key;
        uint32_t slot = hash(key);

        while (true) {
            if (hashtable[slot].key == key) {
                kvs[tid].value = hashtable[slot].value;
                return;
            }
            if (hashtable[slot].key == kEmpty) {
                kvs[tid].value = kEmpty;
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

void lookup_hashtable(
    KeyValue* pHashTable,
    KeyValue* kvs,
    uint32_t num_kvs,
    sycl::queue& qht)
{
    try {
        // Copy this batch of key-value pairs to the device
        KeyValue* device_kvs;
        device_kvs = sycl::malloc_device<KeyValue>(num_kvs, qht);
        auto e1 = qht.memcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs);

        int threadblocksize = 256;

        qht.parallel_for(
            sycl::nd_range<1>(num_kvs, threadblocksize), std::move(e1),
            [=](sycl::nd_item<1> item) {

                gpu_hashtable_lookup(
                    pHashTable,
                    device_kvs,
                    (uint32_t)num_kvs,
                    item);
            }
        );
        qht.wait();

        sycl::free(device_kvs, qht);
    } catch (std::exception const& e) {
        LOG_ERROR("Exception caught, \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception caught, bailing...");
    }
}

// Delete each key in kvs from the hash table, if the key exists
// A deleted key is left in the hash table, but its value is set to kEmpty
// Deleted keys are not reused; once a key is assigned a slot, it never moves
void gpu_hashtable_delete(
    KeyValue* hashtable,
    const KeyValue* kvs,
    unsigned int numkvs,
    sycl::nd_item<1> item)
{
    unsigned int tid = item.get_global_id(0);
    if (tid < numkvs) {
        uint32_t key  = kvs[tid].key;
        uint32_t slot = hash(key);

        while (true) {
            if (hashtable[slot].key == key) {
                hashtable[slot].value = kEmpty;
                return;
            }
            if (hashtable[slot].key == kEmpty) {
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

void delete_hashtable(
    KeyValue* pHashTable,
    const KeyValue* kvs,
    uint32_t num_kvs,
    sycl::queue& qht)
{
    try {
        // Copy the keyvalues to the GPU
        KeyValue* device_kvs;
        device_kvs = sycl::malloc_device<KeyValue>(num_kvs, qht);
        auto e1 = qht.memcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs);

        int threadblocksize = 256;

        qht.parallel_for(
            sycl::nd_range<1>(num_kvs, threadblocksize), std::move(e1),
            [=](sycl::nd_item<1> item) {

                gpu_hashtable_delete(
                    pHashTable,
                    device_kvs,
                    (uint32_t)num_kvs,
                    item) ;
            }
        );
        qht.wait();

        sycl::free(device_kvs, qht);
    } catch (std::exception const& e) {
        LOG_ERROR("Exception caught, \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception caught, bailing...");
    }
}

// Iterate over every item in the hashtable; return non-empty key/values
void gpu_iterate_hashtable(
    KeyValue* pHashTable,
    KeyValue* kvs,
    uint32_t* kvs_size,
    sycl::nd_item<1> item)
{
    unsigned int tid = item.get_global_id(0);
    if (tid < kHashTableCapacity) {
        if (pHashTable[tid].key != kEmpty) {
            uint32_t value = pHashTable[tid].value;
            if (value != kEmpty) {
                // uint32_t size = sycl::atomic<uint32_t>(sycl::global_ptr<uint32_t>(kvs_size)).fetch_add(1);
                uint32_t size = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(kvs_size[0]).fetch_add(1);
                kvs[size] = pHashTable[tid];
            }
        }
    }
}

std::vector<KeyValue> iterate_hashtable(
    KeyValue* pHashTable,
    sycl::queue& qht)
{
    std::vector<KeyValue> kvs;

    try {
        uint32_t* device_num_kvs;
        KeyValue* device_kvs;
        device_num_kvs = sycl::malloc_device<uint32_t>(1, qht);
        device_kvs = sycl::malloc_device<KeyValue>(kNumKeyValues, qht);

        auto e1 = qht.memset(device_num_kvs, 0, sizeof(uint32_t));

        int threadblocksize = 256;

        auto e2 = qht.parallel_for(
            sycl::nd_range<1>(kHashTableCapacity, threadblocksize), std::move(e1),
            [=](sycl::nd_item<1> item) {

                gpu_iterate_hashtable(
                    pHashTable,
                    device_kvs,
                    device_num_kvs,
                    item);
            }
        );

        uint32_t num_kvs;
        qht.memcpy(&num_kvs, device_num_kvs, sizeof(uint32_t), std::move(e2));
        qht.wait();

        kvs.resize(num_kvs);

        qht.memcpy(kvs.data(), device_kvs, sizeof(KeyValue) * num_kvs);
        qht.wait();

        sycl::free(device_kvs, qht);
        sycl::free(device_num_kvs, qht);
    } catch (std::exception const& e) {
        LOG_ERROR("Exception caught, \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception caught, bailing...");
    }

    return kvs;
}

// Free the memory of the hashtable
void destroy_hashtable(
    KeyValue* pHashTable,
    sycl::queue& qht)
{
    sycl::free(pHashTable, qht);
}
