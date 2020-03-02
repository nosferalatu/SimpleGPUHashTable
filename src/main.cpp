#include "windows.h"
#include "stdio.h"
#include "stdint.h"
#include "unordered_set"
#include "unordered_map"
#include "vector"
#include "algorithm"
#include "random"
#include "linearprobing.h"

std::vector<KeyValue> generate_random_keyvalues()
{
    std::random_device rd;
    std::mt19937 gen(rd()); // mersenne_twister_engine
    std::uniform_int_distribution<uint32_t> dis(1, 0xffffffff);    // 0 is reserved for empty keys

    std::vector<KeyValue> kvs;
    kvs.reserve(kNumKeyValues);

    for (int i = 0; i < kNumKeyValues; i++)
    {
        uint32_t rand0 = dis(gen);
        uint32_t rand1 = dis(gen);
        kvs.push_back(KeyValue{ rand0, rand1 });
    }

    return kvs;
}

LARGE_INTEGER start_timer()
{
    LARGE_INTEGER timer;
    QueryPerformanceCounter(&timer);
    return timer;
}

double get_elapsed_time(LARGE_INTEGER start)
{
    LARGE_INTEGER end;
    QueryPerformanceCounter(&end);

    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);

    LARGE_INTEGER elapsed;
    elapsed.QuadPart = end.QuadPart - start.QuadPart;
    elapsed.QuadPart *= 1000000;
    double milliseconds = elapsed.QuadPart / (double)frequency.QuadPart / 1000;

    return milliseconds;
}

void test_unordered_map(std::vector<KeyValue> all_kvs)
{
    LARGE_INTEGER timer = start_timer();

    std::unordered_map<uint32_t, uint32_t> kvs_map;
    for (auto& kv : all_kvs)
    {
        kvs_map[kv.key] = kv.value;
    }

    double milliseconds = get_elapsed_time(timer);
    double seconds = milliseconds / 1000.0f;
    printf("std::unordered_map time : %f ms (%f million keys/second)\n", 
        milliseconds, kNumKeyValues / seconds / 1000000.0f);
}

void test_correctness(std::vector<KeyValue>, std::vector<KeyValue>);

int main()
{
    printf("Testing insertion of %d elements\n", kNumKeyValues);

    // Initialize keyvalue pairs with random numbers
    std::vector<KeyValue> all_kvs = generate_random_keyvalues();

    // Begin test
    LARGE_INTEGER timer = start_timer();

    KeyValue* pHashTable = create_hashtable(kHashTableCapacity);

    // Insert all the elements; each batch is processed concurrently on the GPU
    const uint32_t num_batch_insertions = 16;
    static_assert(kNumKeyValues % num_batch_insertions == 0, "needs even divisor");
    uint32_t num_kvs_in_batch = kNumKeyValues / num_batch_insertions;
    for (int i = 0; i < num_batch_insertions; i++)
    {
        insert_hashtable(pHashTable, all_kvs.data() + i*num_kvs_in_batch, num_kvs_in_batch);
    }

    // Get all the key-values from the hash table
    std::vector<KeyValue> kvs = iterate_hashtable(pHashTable);

    destroy_hashtable(pHashTable);

    // Summarize results
    double milliseconds = get_elapsed_time(timer);
    double seconds = milliseconds / 1000.0f;
    printf("Total time (including memory copies, readback, etc): %f ms (%f million keys/second)\n",
        milliseconds, kNumKeyValues / seconds / 1000000.0f);

    test_unordered_map(all_kvs);

    test_correctness(all_kvs, kvs);

    printf("Success\n");

    return 0;
}
