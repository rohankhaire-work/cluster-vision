#ifndef CLUSTER_CLOUD_CUDA__CLUSTER_CLOUD_CUDA_HPP_
#define CLUSTER_CLOUD_CUDA__CLUSTER_CLOUD_CUDA_HPP_

#include <cuda_runtime_api.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <cmath>

struct VoxelKeyFunctor
{
  float voxel_size;
  float3 min_pt;

  __host__ __device__ VoxelKeyFunctor(float vs, float3 min) : voxel_size(vs), min_pt(min)
  {}

  __host__ __device__ uint64_t
  operator()(const thrust::tuple<float, float, float> &p) const
  {
    int vx = floorf(thrust::get<0>(p) - min_pt.x / voxel_size);
    int vy = floorf(thrust::get<1>(p) - min_pt.y / voxel_size);
    int vz = floorf(thrust::get<2>(p) - min_pt.z / voxel_size);

    // Pack into a unique 64-bit key
    return (static_cast<uint64_t>(vx) & 0x1FFFFF) << 42
           | (static_cast<uint64_t>(vy) & 0x1FFFFF) << 21
           | (static_cast<uint64_t>(vz) & 0x1FFFFF);
  }
};

void executClusteringKernel(float *, float3 *, float, int);

#endif // CLUSTER_CLOUD_CUDA__CLUSTER_CLOUD_CUDA_HPP_
