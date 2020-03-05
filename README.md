![](screenshot.png)

# About

This project shows how to implement a simple GPU hash table. Thanks to the high bandwidth and massive parallelism of
GPU's, the result is a high performance hash table capable of hundreds of millions of operations per second.

The code achieves an average insertion rate of 309 million key/second on my development laptop with an NVIDIA GTX 1060.

The code implements a lock free hash table using linear probing. Concurrent inserts, deletes, and lookups are supported by
this hash table. The hash table works on 32 bit keys and 32 bit values (although 0xffffffff is reserved for both keys
and values). The load factor of the table is set to 50% in the code, and the table size must be a power of two.

Atomic operations are used to insert key/value pairs into the hash table on multiple GPU threads. It uses CUDA for ease
of development, but could easily be ported to HLSL or GLSL. 64 bit keys and/or values could be supported using 64 bit
atomics.

Resizing the hash table is not implemented (it's a *simple* hash table!)

For more information on lock free hash table design, read:
* Preshing on Programming's [The World's Simplest Lock Free Hash
Table](https://preshing.com/20130605/the-worlds-simplest-lock-free-hash-table/)
* Cliff Click's slides on (A Lock-Free Wait-Free Hash Table)[https://web.stanford.edu/class/ee380/Abstracts/070221_LockFreeHash.pdf]

# Prerequisites

* CMake
* CUDA

This has been tested on Windows with Visual Studio Community 2019 on a machine with an NVIDIA GTX 1060.
An easy way to get CMake is to open a Visual Studio command prompt (in Windows, run "x64 Native Tools Command Prompt for
VS 2019"; that will put CMake in your path).

# Cloning

```
git clone https://github.com/nosferalatu/SimpleConcurrentGPUHashTable.git SimpleConcurrentGPUHashTable
```

# Generating Build Files

Run the following commands to generate .sln and .vcxproj's that can be opened in Visual Studio:

```
cd ConcurrentHashTables
md build
cd build
cmake ..
```

You can now open `SimpleConcurrentGPUHashTable.sln` in Visual Studio.

If CMake fails to find CUDA above, then run a CMake generator for 64 bit builds:
```
cmake -G "Visual Studio 16 2019 Win64" ..
```

# Building

You can build within Visual Studio, or from the command line with:

```
cmake --build . --config Release
```
