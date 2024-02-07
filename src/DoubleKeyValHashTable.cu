// Added support for two keys and two values, 

//Create a hash table. For linear probing, this is just an array of keyValues
keyValues* create_hashtable() {
    keyValues* hashtable;
    CUDA_CHECK(cudaMalloc(&hashtable, sizeof(keyValues) * kHashTableCapacity), 
        "Unable to allocate hashtable");

    //Initialize hash table to empty
    static_assert(kEmpty == 0xffffffffffffffff, "memset expected kEmpty=0xffffffffffffffff");
    CUDA_CHECK(cudaMemset(hashtable, 0xff, sizeof(keyValues) * kHashTableCapacity), 
        "Unable memset the hashtable");

    return hashtable;
}

void destroy_hashtable(keyValues* pHashTable) {
    CUDA_CHECK(cudaFree(pHashTable),"Unable to destroy the hashtable");
}

//64 bit Murmur2 hash
__device__ __forceinline__
uint64_t hash(const uint64_t key) {
    const uint32_t seed = 0x9747b28c;
    const uint64_t m = 0xc6a4a7935bd1e995LLU; // A large prime number
    const int r = 47;
  
    uint64_t h = seed ^ (8 * m);

    uint64_t k = key;
  
    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  
    // Finalization
    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h & (kHashTableCapacity - 1); //mask to ensure it falls within table
}

//Combining two keys
__device__ __forceinline__
uint64_t combine_keys(uint32_t key1, uint32_t key2) {
    uint64_t combined_key = key1;
    combined_key = (combined_key << 32) | key2;
    return combined_key;
}

//Lookup keys in the hashtable, and return the values
__global__
void gpu_hashtable_lookup(
    keyValues* hashtable, 
    uint32_t key_1,         // key_1 to lookup
    uint32_t key_2,         // key_2 to lookup
    uint32_t value_1,       // value_1 (result)
    uint32_t value_2,       // value_2 (result)
    int size) {    // num_items

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size) {

        unsigned long long new_key = combine_keys(key_1, key_2);
        uint64_t slot = hash(new_key);

        while(1) {
            if(hashtable[slot].key == new_key) {
                value_1[tid] = hashtable[slot].val1;
                value_2[tid] = hashtable[slot].val2;
                return;
            }
            if(hashtable[slot].key == kEmpty) {
                d_val1[tid] = kEmpty;
                d_val2[tid] = kEmpty;
                return;
            }

            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

// insert into hashTable
__global__
void gpu_hashtable_insert(keyValues* hashtable, uint32_t *key1, uint32_t *key2, uint32_t *value1, uint32_t *value2, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < size) {
        //insert into hashtable here only
        unsigned long long key = combine_keys(key1[tid], key2[tid]);
        
        uint64_t slot = hash(key);

        while(1) {
            unsigned long long prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
            if(prev == kEmpty || prev == key) {
                hashtable[slot].val1 = value1[tid];
                hashtable[slot].val2 = value2[tid];
                return;
            }

            slot = (slot + 1) & (kHashTableCapacity - 1);
        }

    }
}

void insert_hashtable(
    KeyValue* pHashTable, 
    const uint32_t* key_1, const uint32_t* key_2, 
    uint32_t* value_1, const uint32_t* value_2,  
    uint32_t num_items) {

    uint32_t *d_key_1;
    uint32_t *d_key_2;
    uint32_t *d_value_1;
    uint32_t *d_value_2;
    
    // Allocate memory on gpu
    CUDA_CHECK(cudaMalloc(&d_key_1,    sizeof(uint32_t) * num_items), "Failed to allocate d_key_1");
    CUDA_CHECK(cudaMalloc(&d_key_2,    sizeof(uint32_t) * num_items), "Failed to allocate d_key_2");
    CUDA_CHECK(cudaMalloc(&d_value_1,  sizeof(uint32_t) * num_items), "Failed to allocate d_value_1");
    CUDA_CHECK(cudaMalloc(&d_value_2,  sizeof(uint32_t) * num_items), "Failed to allocate d_value_2");

    // Copy the keyvalues to the GPU
    CUDA_CHECK(cudaMemcpy(d_key_1,   key_1,   sizeof(uint32_t) * num_items, cudaMemcpyHostToDevice), 
        "Failed to copy key_1 to gpu");
    CUDA_CHECK(cudaMemcpy(d_key_2,   key_2,   sizeof(uint32_t) * num_items, cudaMemcpyHostToDevice),
        "Failed to copy key_2 to gpu");
    CUDA_CHECK(cudaMemcpy(d_value_1, value_1, sizeof(uint32_t) * num_items, cudaMemcpyHostToDevice),
        "Failed to copy d_value_1 to gpu");
    CUDA_CHECK(cudaMemcpy(d_value_2, value_2, sizeof(uint32_t) * num_items, cudaMemcpyHostToDevice),
        "Failed to copy d_value_2 to gpu");

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);

    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Insert all the keys into the hash table
    int gridsize = (num_items + threadblocksize - 1) / threadblocksize;
    gpu_hashtable_insert<<<gridsize, threadblocksize>>>(
        pHashTable, 
        d_key_1, 
        d_key_2, 
        d_value_1, 
        d_value_2, 
        num_items);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after gpu_hashtable_insert");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("    GPU inserted %d items in %f ms (%f million keys/second)\n", 
        num_items, milliseconds, num_items / (double)seconds / 1000000.0f);

    CUDA_CHECK(cudaFree(d_key_1),   "Failed to free d_key_1");
    CUDA_CHECK(cudaFree(d_key_2),   "Failed to free d_key_2");
    CUDA_CHECK(cudaFree(d_value_1), "Failed to free d_value_1");
    CUDA_CHECK(cudaFree(d_value_2), "Failed to free d_value_2");
}

void lookup_hashtable(KeyValue* pHashTable, 
    const uint32_t* key_1,      // cpu key_1
    const uint32_t* key_2,      // cpu key_2
    const uint32_t* value_1,    // cpu value_1 (result)
    const uint32_t* value_2,    // cpu value_2 (result)
    uint32_t num_items) {

    uint32_t *d_key_1;
    uint32_t *d_key_2;
    uint32_t *d_value_1;
    uint32_t *d_value_2;
    
    // Allocate memory on gpu
    CUDA_CHECK(cudaMalloc(&d_key_1,    sizeof(uint32_t) * num_items), "Failed to allocate d_key_1");
    CUDA_CHECK(cudaMalloc(&d_key_2,    sizeof(uint32_t) * num_items), "Failed to allocate d_key_2");
    CUDA_CHECK(cudaMalloc(&d_value_1,  sizeof(uint32_t) * num_items), "Failed to allocate d_value_1");
    CUDA_CHECK(cudaMalloc(&d_value_2,  sizeof(uint32_t) * num_items), "Failed to allocate d_value_2");

    // Copy the keyvalues to the GPU
    CUDA_CHECK(cudaMemcpy(d_key_1, key_1, sizeof(uint32_t) * num_items, cudaMemcpyHostToDevice), 
        "Failed to copy key_1 to gpu");
    CUDA_CHECK(cudaMemcpy(d_key_2, key_2, sizeof(uint32_t) * num_items, cudaMemcpyHostToDevice),
        "Failed to copy key_2 to gpu");

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);

    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpu_hashtable_lookup<<<gridsize, threadblocksize>>>(
        pHashTable,
        d_key_1, 
        d_key_2, 
        d_value_1,
        d_value_2, 
        h_size);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after gpu_hashtable_insert");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("    GPU looked_up %d items in %f ms (%f million keys/second)\n", 
        num_items, milliseconds, num_items / (double)seconds / 1000000.0f);

    // copy back results to cpu
    CUDA_CHECK(cudaMemcpy(value_1, d_value_1, sizeof(uint32_t) * num_items, cudaMemcpyHostToDevice), 
        "Failed to copy back d_value_1 back to CPU");
    CUDA_CHECK(cudaMemcpy(value_2, d_value_2, sizeof(uint32_t) * num_items, cudaMemcpyHostToDevice), 
        "Failed to copy back d_value_2 back to CPU");

    CUDA_CHECK(cudaFree(d_key_1),   "Failed to free d_key_1");
    CUDA_CHECK(cudaFree(d_key_2),   "Failed to free d_key_2");
    CUDA_CHECK(cudaFree(d_value_1), "Failed to free d_value_1");
    CUDA_CHECK(cudaFree(d_value_2), "Failed to free d_value_2");
}