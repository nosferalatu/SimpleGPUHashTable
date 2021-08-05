#include "stdio.h"
#include "stdint.h"
#include "unordered_set"
#include "unordered_map"
#include "vector"
#include "algorithm"
#include "random"
#include "linearprobing.h"

void test_correctness(std::vector<KeyValue> insert_kvs, std::vector<KeyValue> delete_kvs, std::vector<KeyValue> kvs)
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

    printf("Building unordered_map from original list...\n");
    std::unordered_map<uint32_t, std::vector<uint32_t>> all_kvs_map;
    for (int i = 0; i < insert_kvs.size(); i++)
    {
        if (i % 10000000 == 0)
            printf("    Inserting %d/%d\n", i, (uint32_t)insert_kvs.size());

        auto iter = all_kvs_map.find(insert_kvs[i].key);
        if (iter == all_kvs_map.end())
        {
            all_kvs_map[insert_kvs[i].key] = std::vector<uint32_t>({ insert_kvs[i].value });
        }
        else
        {
            iter->second.push_back(insert_kvs[i].value);
        }
    }

    for (int i = 0; i < delete_kvs.size(); i++)
    {
        if (i % 10000000 == 0)
            printf("    Deleting %d/%d\n", i, (uint32_t)delete_kvs.size());

        auto iter = all_kvs_map.find(delete_kvs[i].key);
        if (iter != all_kvs_map.end())
        {
            all_kvs_map.erase(iter);
        }
    }

    if (unique_keys.size() != all_kvs_map.size())
    {
        printf("# of unique keys in hashtable is incorrect\n");
        exit(-1);
    }

    printf("Testing that each key/value in hashtable is in the original list...\n");
    for (uint32_t i = 0; i < kvs.size(); i++)
    {
        if (i % 10000000 == 0)
            printf("    Verifying %d/%d\n", i, (uint32_t)kvs.size());

        auto iter = all_kvs_map.find(kvs[i].key);
        if (iter == all_kvs_map.end())
        {
            printf("Hashtable key not found in original list\n");
            exit(-1);
        }

        std::vector<uint32_t>& values = iter->second;
        if (std::find(values.begin(), values.end(), kvs[i].value) == values.end())
        {
            printf("Hashtable value not found in original list\n");
            exit(-1);
        }
    }

    printf("Deleting std::unordered_map and std::unique_set...\n");

    return;
}
