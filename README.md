# hashtable

hashtable implements a simple hash table in GPU (original CUDA source code is from [here](https://github.com/nosferalatu/SimpleGPUHashTable)).


## SYCL version

- The CUDA code was migrated using Intel DPCT, and then the resulting code was modified to remove the DPCT headers.
- Timing code was later added for performance measurement purpose.
- The same SYCL code runs on Intel GPUs & CPUs as well as NVIDIA (tested on A100 and H100) and AMD (tested on MI100 and MI250) GPUs.

# Build Instructions
- icpx compiler mentioned below is included in oneAPI Base Toolkit available [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).
- clang++ compiler mentioned below is available [here](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md).
## To build for SYCL

For Intel GPU -  
First, source icpx compiler. Then,

```
mkdir build
cd build
CXX=icpx cmake -DGPU_AOT=pvc ..
make -sj
```
Note:
- To enable AOT compilation, please use the flag `-DGPU_AOT=pvc` for PVC.

For AMD GPU -  
First source clang++ compiler. Then,
```
mkdir build
cd build
CXX=clang++ cmake -DUSE_AMDHIP_BACKEND=gfx90a ..
make -sj
```
Note:
- We use the flag `-DUSE_AMDHIP_BACKEND=gfx90a` for MI250. Use the correct value for your GPU.

For NVIDIA GPU -  
First source clang++ compiler. Then,
```
mkdir build
cd build
CXX=clang++ cmake -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=80 ..
make -sj
```
Note:
- We use the flag `-DUSE_SM=80` for A100 or `-DUSE_SM=90` for H100.

# Run instructions

After building, to run the workload, cd into the build folder. Then

```
./hashtable_sycl
```
By default a verification is done and that takes some time. To skip verification:
```
./hashtable_sycl --no-verify
```
# Output

Output gives number of keys per second.
