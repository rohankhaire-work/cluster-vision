#include "cluster_vision/plane_segmentation_cuda.hpp"

// Device function: compute cross product
__device__ float3 cross(const float3 &a, const float3 &b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Device function: vector subtraction
__device__ float3 subtract(const float3 &a, const float3 &b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

// Device function: vector length
__device__ float length(const float3 &v)
{
  return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Device function: check colinearity (returns true if points are colinear)
__device__ bool
areColinear(const float3 &p1, const float3 &p2, const float3 &p3, float epsilon = 1e-6f)
{
  float3 v1 = subtract(p2, p1);
  float3 v2 = subtract(p3, p1);
  float3 cross_prod = cross(v1, v2);
  return length(cross_prod) < epsilon;
}

// Device function: compute plane parameters from 3 points
__device__ void
computePlane(const float3 &p1, const float3 &p2, const float3 &p3, float4 *plane)
{
  float3 v1 = subtract(p2, p1);
  float3 v2 = subtract(p3, p1);
  float3 normal = cross(v1, v2);
  float norm_len = length(normal);
  if(norm_len < 1e-6f)
  {
    plane->x = plane->y = plane->z = plane->w = 0.0f; // invalid plane
    return;
  }
  normal.x /= norm_len;
  normal.y /= norm_len;
  normal.z /= norm_len;
  plane->x = normal.x;
  plane->y = normal.y;
  plane->z = normal.z;
  plane->w = -(normal.x * p1.x + normal.y * p1.y + normal.z * p1.z);
}

// RANSAC kernel: compute inliers per candidate plane with colinearity check
__global__ void
ransacPlaneKernel(float *points, int num_points, int3 *samples, int num_samples,
                  float threshold, int *inlier_counts, float4 *plane_params)
{
  int candidate_id = blockIdx.x;
  if(candidate_id >= num_samples)
    return;

  int tid = threadIdx.x;
  int3 sample = samples[candidate_id];

  // Get the point in the cloud
  float3 p1 = make_float3(points[sample.x * 4 + 0], points[sample.x * 4 + 1],
                          points[sample.x * 4 + 2]);

  float3 p2 = make_float3(points[sample.y * 4 + 0], points[sample.y * 4 + 1],
                          points[sample.y * 4 + 2]);

  float3 p3 = make_float3(points[sample.z * 4 + 0], points[sample.z * 4 + 1],
                          points[sample.z * 4 + 2]);

  // Check colinearity
  if(areColinear(p1, p2, p3))
  {
    if(tid == 0)
      inlier_counts[candidate_id] = 0;
    return;
  }

  float4 plane;
  computePlane(p1, p2, p3, &plane);
  extern __shared__ int local_inliers[];
  int local_count = 0;

  for(int i = tid; i < num_points; i += blockDim.x)
  {
    auto point = (reinterpret_cast<float4 *>(points)[i]);
    float dist
      = fabsf(plane.x * point.x + plane.y * point.y + plane.z * point.z + plane.w);

    if(dist < 0.001)
      local_count++;
  }
  local_inliers[tid] = local_count;
  __syncthreads();

  for(int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if(tid < s)
      local_inliers[tid] += local_inliers[tid + s];
    __syncthreads();
  }

  if(tid == 0)
  {
    inlier_counts[candidate_id] = local_inliers[0];
    plane_params[candidate_id] = plane;
  }
}

// Kernel to extract inliers for the best plane
__global__ void extractInliersForBestPlane(const float4 *points, int num_points,
                                           const float4 *best_plane_ptr, float threshold,
                                           int *out_indices, int *out_count)
{
  float4 best_plane = *best_plane_ptr;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid >= num_points)
    return;

  float4 p = points[tid];
  float dist
    = fabsf(best_plane.x * p.x + best_plane.y * p.y + best_plane.z * p.z + best_plane.w);

  if(dist < threshold)
  {
    int idx = atomicAdd(out_count, 1);
    out_indices[idx] = tid;
  }
}

void executePlaneSegmentationKernel(float *points, float4 *best_plane,
                                    int *inlier_indices, int *indices_count,
                                    float dist_threshold, size_t num_samples,
                                    size_t num_points, unsigned int seed)
{
  // RNG for RANSAC
  thrust::device_vector<int3> d_samples(num_samples);
  thrust::counting_iterator<int> index_sequence_begin(0);
  thrust::transform(index_sequence_begin, index_sequence_begin + num_samples,
                    d_samples.begin(), RandomTripletGenerator(num_points, seed));

  int threads = 256;
  int blocks = num_samples;
  size_t shared_mem_size = threads * sizeof(int);

  thrust::device_vector<int> inlier_counts(num_samples);
  thrust::device_vector<float4> plane_params(num_samples);
  ransacPlaneKernel<<<blocks, threads, shared_mem_size>>>(
    points, num_points, thrust::raw_pointer_cast(d_samples.data()), num_samples,
    dist_threshold, thrust::raw_pointer_cast(inlier_counts.data()),
    thrust::raw_pointer_cast(plane_params.data()));

  // Find best plane index
  auto max_it = thrust::max_element(inlier_counts.begin(), inlier_counts.end());
  int best_idx = max_it - inlier_counts.begin();

  // Get best plane params from GPU
  cudaMemcpy(best_plane, thrust::raw_pointer_cast(plane_params.data()) + best_idx,
             sizeof(float4), cudaMemcpyDeviceToHost);

  // Extract inliers
  cudaMemset(indices_count, 0, sizeof(int));
  blocks = (num_points + threads - 1) / threads;
  extractInliersForBestPlane<<<blocks, threads>>>(
    reinterpret_cast<const float4 *>(points), num_points, best_plane, dist_threshold,
    inlier_indices, indices_count);

  cudaDeviceSynchronize();
}
