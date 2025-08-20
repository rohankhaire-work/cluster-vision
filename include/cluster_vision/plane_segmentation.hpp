#ifndef PLANE_SEGMENTATION__PLANE_SEGMENTATION_HPP_
#define PLANE_SEGMENTATION__PLANE_SEGMENTATION_HPP_

#include "cluster_vision/plane_segmentation_cuda.hpp"

#include <pcl/common/common.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/kdtree.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <memory>

struct CloudParams
{
  size_t total_size, point_size;
  float distance_threshold;
  size_t num_samples, num_points;
};

class PlaneSegmentation
{
public:
  PlaneSegmentation(const std::shared_ptr<CloudParams> &);
  ~PlaneSegmentation();

  void extractPlaneCUDA(const pcl::PointCloud<pcl::PointXYZ>::Ptr &,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr &, std::vector<float> &);

private:
  std::shared_ptr<CloudParams> cloud_params_;

  float *d_points_, *h_points_;
  int *h_indices_count_, *h_inlier_indices_, *d_inlier_indices_, *d_indices_count_;
  float4 *h_best_plane_, *d_best_plane_;
  cudaStream_t stream_;

  void allocateMemory();
  void clearAllocatedMemory();
  void getInlierCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr &, int *, int);
};

#endif // PLANE_SEGMENTATION__PLANE_SEGMENTATION_HPP_
