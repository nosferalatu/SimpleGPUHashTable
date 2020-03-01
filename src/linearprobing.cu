#include "stdio.h"
#include "stdint.h"
#include "vector"
#include "linearprobing.h"

// 32 bit Murmur3 hash
__device__ uint32_t hash(uint32_t k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (kHashTableCapacity-1);
}

__global__ void gpu_hashtable_insert(KeyValue* hashtable, const KeyValue* kvs, unsigned int numkvs)
{
    unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadid < numkvs)
    {
        uint32_t key = kvs[threadid].key;
        uint32_t value = kvs[threadid].value;
        uint32_t slot = hash(key);

        while (true)
        {
            uint32_t prev = atomicCAS(&hashtable[slot].key, 0, key);
            if (prev == 0 || prev == key)
            {
                hashtable[slot].value = value;
                break;
            }

            slot = (slot + 1) & (kHashTableCapacity-1);
        }
    }
}
 
KeyValue* create_hashtable(uint32_t capacity)
{
    // Allocate memory
    KeyValue* hashtable;
    cudaMalloc(&hashtable, sizeof(KeyValue)*kHashTableCapacity);
    
    // Initialize hash table to empty
    cudaMemset(hashtable, 0, sizeof(KeyValue)*kHashTableCapacity);

    return hashtable;
}

void insert_hashtable(KeyValue* pHashTable, const KeyValue* kvs, uint32_t num_kvs)
{
    // Copy the keyvalues to the GPU
    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs);
    cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);

    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Hash table insertion
    int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;
    gpu_hashtable_insert<<<gridsize, threadblocksize>>>(pHashTable, device_kvs, (uint32_t)num_kvs);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("    GPU inserted %d items in %f ms (%f million keys/second)\n", num_kvs, milliseconds, num_kvs / (double)seconds / 1000000.0f);

    cudaFree(device_kvs);
}

std::vector<KeyValue> iterate_hashtable(KeyValue* pHashTable)
{
    KeyValue* pHostHashTable;
    cudaHostAlloc(&pHostHashTable, sizeof(KeyValue) * kHashTableCapacity, 0);
    cudaMemcpy(pHostHashTable, pHashTable, sizeof(KeyValue) * kHashTableCapacity, cudaMemcpyDeviceToHost);

    std::vector<KeyValue> kvs;
    kvs.reserve(kHashTableCapacity);

    for (uint32_t i = 0; i < kHashTableCapacity; i++)
    {
        if (pHostHashTable[i].key != 0)
        {
            kvs.push_back(pHostHashTable[i]);
        }
    }

    cudaFree(pHostHashTable);

    return kvs;
}

void destroy_hashtable(KeyValue* pHashTable)
{
    cudaFree(pHashTable);
}
