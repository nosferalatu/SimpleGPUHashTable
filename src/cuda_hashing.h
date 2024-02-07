#ifndef CUDA_HASHING_H_
#define CUDA_HASHING_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <stdint.h>
#include <set>

struct keyValues
{
	unsigned long long key;
	uint32_t val1;
	uint32_t val2;
};

const uint64_t kHashTableCapacity = 128 * 1024 * 1024;

const uint64_t kEmpty = 0xffffffffffffffff;

keyValues* create_hashtable();
void insert_hashtable(keyValues* pHashTable, const std::vector<int>& v1, const std::vector<int>& v2, const std::vector<int>& v3, const std::vector<int>& v4);

void lookup_hashtable(keyValues* hashtable, const std::vector<int>& key1, const std::vector<int>& key2, std::vector<long unsigned int>& value1, std::vector<long unsigned int>& value2);

void destroy_hashtable(keyValues* pHashTable);

#endif  // CUDA_HASHING_H_


