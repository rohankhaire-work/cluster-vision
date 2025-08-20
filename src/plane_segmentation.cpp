#include "cluster_vision/plane_segmentation.hpp"

PlaneSegmentation::PlaneSegmentation(const std::shared_ptr<CloudParams> &cloud_params)
{
  cloud_params_ = cloud_params;
  allocateMemory();
}

PlaneSegmentation::~PlaneSegmentation() { clearAllocatedMemory(); }

void PlaneSegmentation::extractPlaneCUDA(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                         pcl::PointCloud<pcl::PointXYZ>::Ptr &inlier_cloud,
                                         std::vector<float> &best_plane)
{
  float *cloud_xyz = reinterpret_cast<float *>(cloud->points.data());
  // Copy to host memory first then to device memory
  std::memcpy(h_points_, cloud_xyz, cloud_params_->total_size);
  cudaMemcpyAsync(d_points_, h_points_, cloud_params_->total_size, cudaMemcpyHostToDevice,
                  stream_);

  // Perform Plane Segmentation on GPU
  unsigned int seed = static_cast<unsigned int>(
    std::chrono::system_clock::now().time_since_epoch().count());
  executePlaneSegmentationKernel(d_points_, d_best_plane_, d_inlier_indices_,
                                 d_indices_count_, cloud_params_->distance_threshold,
                                 cloud_params_->num_samples, cloud_params_->num_points,
                                 seed);

  // Copy data back to host containers
  cudaMemcpyAsync(h_indices_count_, d_indices_count_, sizeof(int), cudaMemcpyDeviceToHost,
                  stream_);
  cudaMemcpyAsync(h_inlier_indices_, d_inlier_indices_,
                  cloud_params_->num_points * sizeof(int), cudaMemcpyDeviceToHost,
                  stream_);
  cudaMemcpyAsync(h_best_plane_, d_best_plane_, sizeof(float4), cudaMemcpyDeviceToHost,
                  stream_);

  cudaStreamSynchronize(stream_);

  best_plane = {h_best_plane_->x, h_best_plane_->y, h_best_plane_->z, h_best_plane_->w};
  int inliers_count = h_indices_count_[0];
  inlier_cloud->points.resize(inliers_count);
  getInlierCloud(cloud, inlier_cloud, h_inlier_indices_, inliers_count);
}

void PlaneSegmentation::getInlierCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                       pcl::PointCloud<pcl::PointXYZ>::Ptr &inlier_cloud,
                                       int *inlier_indices, int inlier_count)
{
  for(int i = 0; i < inlier_count; ++i)
  {
    int idx = inlier_indices[i];
    if(idx >= 0 && idx < static_cast<int>(cloud->points.size()))
    {
      inlier_cloud->points[i] = cloud->points[idx];
    }
  }

  inlier_cloud->width = inlier_cloud->points.size();
  inlier_cloud->height = 1;
  inlier_cloud->is_dense = false;
}

void PlaneSegmentation::allocateMemory()
{
  cudaMallocHost(reinterpret_cast<void **>(&h_points_), cloud_params_->total_size);
  cudaMallocHost(reinterpret_cast<void **>(&h_indices_count_), sizeof(int));
  cudaMallocHost(reinterpret_cast<void **>(&h_best_plane_), sizeof(float4));
  cudaMallocHost(reinterpret_cast<void **>(&h_inlier_indices_),
                 cloud_params_->num_points * sizeof(int));

  cudaMalloc(reinterpret_cast<void **>(&d_points_), cloud_params_->total_size);
  cudaMalloc(reinterpret_cast<void **>(&d_inlier_indices_),
             cloud_params_->num_points * sizeof(int));
  cudaMalloc(reinterpret_cast<void **>(&d_indices_count_), sizeof(int));
  cudaMalloc(reinterpret_cast<void **>(&d_best_plane_), sizeof(float4));
  cudaStreamCreate(&stream_);
}

void PlaneSegmentation::clearAllocatedMemory()
{
  if(d_points_)
  {
    cudaFree(d_points_);
    d_points_ = nullptr;
  }
  if(d_inlier_indices_)
  {
    cudaFree(d_inlier_indices_);
    d_inlier_indices_ = nullptr;
  }
  if(d_indices_count_)
  {
    cudaFree(d_indices_count_);
    d_indices_count_ = nullptr;
  }
  if(d_best_plane_)
  {
    cudaFree(d_best_plane_);
    d_best_plane_ = nullptr;
  }

  if(h_points_)
  {
    cudaFreeHost(h_points_);
    h_points_ = nullptr;
  }
  if(h_indices_count_)
  {
    cudaFreeHost(h_indices_count_);
    h_indices_count_ = nullptr;
  }
  if(h_inlier_indices_)
  {
    cudaFreeHost(h_inlier_indices_);
    h_inlier_indices_ = nullptr;
  }
  if(h_best_plane_)
  {
    cudaFreeHost(h_best_plane_);
    h_best_plane_ = nullptr;
  }

  if(stream_)
  {
    cudaStreamDestroy(stream_);
  }
}
