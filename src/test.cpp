#include "windows.h"
#include "stdio.h"
#include "stdint.h"
#include "unordered_set"
#include "unordered_map"
#include "vector"
#include "algorithm"
#include "random"
#include "linearprobing.h"

void test_correctness(std::vector<KeyValue> all_kvs, std::vector<KeyValue> kvs)
{
    printf("Testing that there are no duplicate keys...\n");
    std::unordered_set<uint32_t> unique_keys;
    for (uint32_t i = 0; i < kvs.size(); i++)
    {
        if (i % 10000000 == 0)
            printf("    Verifying %d/%d\n", i, (uint32_t)kvs.size());

        KeyValue* node = &kvs[i];
        if (unique_keys.find(node->key) != unique_keys.end())
        {
            printf("Duplicate key found in GPU hash table at slot %d\n", i);
            exit(-1);
        }
        unique_keys.insert(node->key);
    }

    printf("Building unordered_map from insertion list...\n");
    std::unordered_map<uint32_t, std::vector<uint32_t>> all_kvs_map;
    for (int i = 0; i < all_kvs.size(); i++)
    {
        if (i % 10000000 == 0)
            printf("    Building %d/%d\n", i, (uint32_t)all_kvs.size());

        auto iter = all_kvs_map.find(all_kvs[i].key);
        if (iter == all_kvs_map.end())
        {
            all_kvs_map[all_kvs[i].key] = std::vector<uint32_t>({ all_kvs[i].value });
        }
        else
        {
            iter->second.push_back(all_kvs[i].value);
        }
    }

    if (unique_keys.size() != all_kvs_map.size())
    {
        printf("# of unique keys in hashtable is incorrect\n");
        exit(-1);
    }

    printf("Testing that each key/value in hashtable is in the insertion list...\n");
    for (uint32_t i = 0; i < kvs.size(); i++)
    {
        if (i % 10000000 == 0)
            printf("    Verifying %d/%d\n", i, (uint32_t)kvs.size());

        auto iter = all_kvs_map.find(kvs[i].key);
        if (iter == all_kvs_map.end())
        {
            printf("Hashtable key not found in insertion list\n");
            exit(-1);
        }

        std::vector<uint32_t>& values = iter->second;
        if (std::find(values.begin(), values.end(), kvs[i].value) == values.end())
        {
            printf("Hashtable value not found in insertion list\n");
            exit(-1);
        }
    }

    printf("Deleting std::unordered_map and std::unique_set...\n");

    return;
}
