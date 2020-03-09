#include "algorithm"
#include "random"
#include "stdint.h"
#include "stdio.h"
#include "unordered_map"
#include "unordered_set"
#include "vector"
#include "chrono"
#include "linearprobing.h"

// Create random keys/values in the range [0, kEmpty)
// kEmpty is used to indicate an empty slot
std::vector<KeyValue> generate_random_keyvalues(uint32_t numkvs)
{
    std::random_device rd;
    std::mt19937 gen(rd());  // mersenne_twister_engine
    std::uniform_int_distribution<uint32_t> dis(0, kEmpty - 1);

    std::vector<KeyValue> kvs;
    kvs.reserve(numkvs);

    for (uint32_t i = 0; i < numkvs; i++)
    {
        uint32_t rand0 = dis(gen);
        uint32_t rand1 = dis(gen);
        kvs.push_back(KeyValue{rand0, rand1});
    }

    return kvs;
}

// return numshuffledkvs random items from kvs
std::vector<KeyValue> shuffle_keyvalues(std::vector<KeyValue> kvs, uint32_t numshuffledkvs)
{
    std::random_shuffle(kvs.begin(), kvs.end());

    std::vector<KeyValue> shuffled_kvs;
    shuffled_kvs.resize(numshuffledkvs);

    std::copy(kvs.begin(), kvs.begin() + numshuffledkvs, shuffled_kvs.begin());

    return shuffled_kvs;
}

using Time = std::chrono::time_point<std::chrono::steady_clock>;

Time start_timer() 
{
    return std::chrono::high_resolution_clock::now();
}

double get_elapsed_time(Time start) 
{
    Time end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> d = end - start;
    std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(d);
    return us.count() / 1000.0f;
}

void test_unordered_map(std::vector<KeyValue> insert_kvs, std::vector<KeyValue> delete_kvs) 
{
    Time timer = start_timer();

    printf("Timing std::unordered_map...\n");

    {
        std::unordered_map<uint32_t, uint32_t> kvs_map;
        for (auto& kv : insert_kvs) 
        {
            kvs_map[kv.key] = kv.value;
        }
        for (auto& kv : delete_kvs)
        {
            auto i = kvs_map.find(kv.key);
            if (i != kvs_map.end())
                kvs_map.erase(i);
        }
    }

    double milliseconds = get_elapsed_time(timer);
    double seconds = milliseconds / 1000.0f;
    printf("Total time for std::unordered_map: %f ms (%f million keys/second)\n", 
        milliseconds, kNumKeyValues / seconds / 1000000.0f);
}

void test_correctness(std::vector<KeyValue>, std::vector<KeyValue>, std::vector<KeyValue>);

int main() 
{
    while (true)
    {
        // Initialize keyvalue pairs with random numbers
        std::vector<KeyValue> insert_kvs = generate_random_keyvalues(kNumKeyValues);
        std::vector<KeyValue> delete_kvs = shuffle_keyvalues(insert_kvs, kNumKeyValues / 2);

        printf("Testing insertion/deletion of %d/%d elements into GPU hash table...\n",
            (uint32_t)insert_kvs.size(), (uint32_t)delete_kvs.size());

        // Begin test
        Time timer = start_timer();

        KeyValue* pHashTable = create_hashtable(kHashTableCapacity);

        // Insert items into the hash table
        const uint32_t num_insert_batches = 16;
        uint32_t num_inserts_per_batch = (uint32_t)insert_kvs.size() / num_insert_batches;
        for (uint32_t i = 0; i < num_insert_batches; i++)
        {
            insert_hashtable(pHashTable, insert_kvs.data() + i * num_inserts_per_batch, num_inserts_per_batch);
        }

        // Delete items from the hash table
        const uint32_t num_delete_batches = 8;
        uint32_t num_deletes_per_batch = (uint32_t)delete_kvs.size() / num_delete_batches;
        for (uint32_t i = 0; i < num_delete_batches; i++)
        {
            delete_hashtable(pHashTable, delete_kvs.data() + i * num_deletes_per_batch, num_deletes_per_batch);
        }

        // Get all the key-values from the hash table
        std::vector<KeyValue> kvs = iterate_hashtable(pHashTable);

        destroy_hashtable(pHashTable);

        // Summarize results
        double milliseconds = get_elapsed_time(timer);
        double seconds = milliseconds / 1000.0f;
        printf("Total time (including memory copies, readback, etc): %f ms (%f million keys/second)\n", milliseconds,
            kNumKeyValues / seconds / 1000000.0f);

        test_unordered_map(insert_kvs, delete_kvs);

        test_correctness(insert_kvs, delete_kvs, kvs);

        printf("Success\n");
    }

    return 0;
}
