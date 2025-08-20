#ifndef PLANE_SEGMENTATION_CUDA__PLANE_SEGMENTATION_CUDA_HPP_
#define PLANE_SEGMENTATION_CUDA__PLANE_SEGMENTATION_CUDA_HPP_

#include <cmath>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/random.h>

// Functor to generate distinct random triplets using Thrust RNG
struct RandomTripletGenerator
{
  int num_points;
  unsigned int seed;

  __host__ __device__ RandomTripletGenerator(int np, unsigned int s)
      : num_points(np), seed(s)
  {}

  __device__ int3 operator()(int idx)
  {
    thrust::default_random_engine rng(seed);
    thrust::uniform_int_distribution<int> dist(0, num_points - 1);

    rng.discard(idx * 3);

    int i1, i2, i3;
    do
    {
      i1 = dist(rng);
      i2 = dist(rng);
      i3 = dist(rng);
    } while(i1 == i2 || i2 == i3 || i1 == i3);

    return {i1, i2, i3};
  }
};

void executePlaneSegmentationKernel(float *, float4 *, int *, int *, float, size_t,
                                    size_t, unsigned int);

#endif // PLANE_SEGMENTATION_CUDA__PLANE_SEGMENTATION_CUDA_HPP_
