#ifndef CLUSTER_VISION__CLUSTER_VISION_HPP_
#define CLUSTER_VISION__CLUSTER_VISION_HPP_

#include "cluster_vision/plane_segmentation.hpp"

#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <string>

class ClusterVision : public rclcpp::Node
{
public:
  ClusterVision();
  ~ClusterVision();

private:
  // Params
  std::string lidar_topic_, depth_topic_, base_frame_, camera_frame_;
  bool downsample_;
  int max_iters_, num_samples_;
  double distance_threshold_, norm_dist_wt_;
  float leaf_size_;

  // Variables
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
  int point_size_ = 16; // We have XYZ pointcloud
  size_t total_size_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<PlaneSegmentation> plane_segmentation_;
  std::shared_ptr<CloudParams> cloud_params_;

  // Subscriber
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Publisher
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr plane_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr line_pub_;

  // Callback
  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &);
  void timerCallback();

  // Functions
  void setCloudParams();
  std::pair<pcl::ModelCoefficients, pcl::PointIndices>
  extractPlaneCPU(pcl::PointCloud<pcl::PointXYZ>::Ptr, double, int, double);
  void publishPlane(const std::pair<pcl::ModelCoefficients, pcl::PointIndices> &,
                    const pcl::PointCloud<pcl::PointXYZ> &);

  void publishPlane(const std::vector<float> &, const pcl::PointCloud<pcl::PointXYZ> &);
};

#endif // CLUSTER_VISION__CLUSTER_VISION_HPP_
