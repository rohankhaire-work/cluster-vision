#include "cluster_vision/cluster_vision_node.hpp"

ClusterVision::ClusterVision() : Node("cluster_vision_node")
{
  // Set Parameters
  lidar_topic_ = declare_parameter<std::string>("lidar_topic", "");
  depth_topic_ = declare_parameter<std::string>("depth_topic", "");
  base_frame_ = declare_parameter<std::string>("base_frame", "");
  camera_frame_ = declare_parameter<std::string>("camera_frame", "");
  downsample_ = declare_parameter("downsample", true);
  leaf_size_ = static_cast<float>(declare_parameter("leaf_size", 0.05));
  distance_threshold_ = declare_parameter("distance_threshold", 0.5);
  num_samples_ = declare_parameter("num_samples", 1024);

  // Initialize PCL pointcloud
  cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

  // Initialize tf2 for transforms
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

  // Subscribers
  cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    lidar_topic_, 1,
    std::bind(&ClusterVision::cloudCallback, this, std::placeholders::_1));

  timer_ = this->create_wall_timer(std::chrono::milliseconds(50),
                                   std::bind(&ClusterVision::timerCallback, this));
#ifdef GPU_ACCELERATION
  // Initalize plane segmentation
  setCloudParams();
  plane_segmentation_ = std::make_unique<PlaneSegmentation>(cloud_params_);
#endif

#ifdef VISUALIZE_FEATURES
  plane_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("plane_viz", 10);
#endif
}

ClusterVision::~ClusterVision()
{
  timer_->cancel();
  plane_segmentation_.reset();
}

void ClusterVision::setCloudParams()
{
  // XYZ Cloud with padding
  int point_channels = 4;
  int channel_size = 4;

  cloud_params_->num_samples = num_samples_;
  cloud_params_->num_points = cloud_->points.size();
  cloud_params_->total_size = cloud_->points.size() * point_channels * channel_size;
  cloud_params_->distance_threshold = distance_threshold_;
}

void ClusterVision::cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
{
  // Convert from ROS to PCL
  pcl::fromROSMsg(*msg, *cloud_);
  // Downsample Pointcloud if true
  if(downsample_)
  {
    // Remove NaNs
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud_, *cloud_, indices);

    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(cloud_);
    voxel_filter.setLeafSize(leaf_size_, leaf_size_, leaf_size_);

    voxel_filter.filter(*cloud_);
  }
}

void ClusterVision::timerCallback()
{
  // Check if the image and pointcloud exists
  if(cloud_->empty())
  {
    RCLCPP_WARN(this->get_logger(), "Pointcloud is missing in Feature Extractor");
    return;
  }

  // Extract Planes
  auto start_time = std::chrono::steady_clock::now();
#ifdef GPU_ACCELERATION
  std::vector<float> best_plane;
  pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  plane_segmentation_->extractPlaneCUDA(cloud_, inlier_cloud, best_plane);
#else
  std::pair<pcl::ModelCoefficients, pcl::PointIndices> planes = extractPlaneCPU(
    cloud_, max_plane_count_, norm_dist_wt_, max_iters_, distance_threshold_);
#endif
  auto end_time = std::chrono::steady_clock::now();
  auto duration_ms
    = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  RCLCPP_INFO(this->get_logger(), "Plane extraction took %ld ms", duration_ms);

  // Clustering

#if defined(GPU_ACCELERATION) && defined(VISUALIZE_FEATURES)
  publishPlanes(best_plane, inlier_cloud);
#elif defined(VISUALIZE_FEATURES)
  publishPlanes(planes, *cloud_);
#endif
}

std::pair<pcl::ModelCoefficients, pcl::PointIndices>
ClusterVision::extractPlaneCPU(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                               double normal_distance_weight, int max_iterations,
                               double distance_threshold)
{
  std::pair<pcl::ModelCoefficients, pcl::PointIndices> plane;
  pcl::ExtractIndices<pcl::PointXYZ> extract;

  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(max_iterations);
  seg.setDistanceThreshold(distance_threshold);
  seg.setInputCloud(cloud);

  pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);

  seg.segment(*inliers_plane, *coefficients_plane);

  return {*coefficients_plane, *inliers_plane};
}

void ClusterVision::publishPlane(const std::vector<float> &best_plane,
                                 const pcl::PointCloud<pcl::PointXYZ> &inlier_cloud)
{
  visualization_msgs::msg::MarkerArray marker_array;
  visualization_msgs::msg::Marker plane_marker;
  plane_marker.header.frame_id
    = base_frame_; // your fixed frame id, e.g. "map" or "base_link"
  plane_marker.header.stamp = this->now();
  plane_marker.ns = "detected_planes";
  plane_marker.id = 0; // unique id per marker
  plane_marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
  plane_marker.action = visualization_msgs::msg::Marker::ADD;
  plane_marker.scale.x = 1.0;
  plane_marker.scale.y = 1.0;
  plane_marker.scale.z = 1.0;
  plane_marker.color.r = 0.0f;
  plane_marker.color.g = 0.5f;
  plane_marker.color.b = 1.0f;
  plane_marker.color.a = 0.5f; // semi-transparent

  // Compute bounding box corners of the inliers
  pcl::PointXYZ min_pt, max_pt;
  pcl::getMinMax3D(inlier_cloud, min_pt, max_pt);

  // Lambda to project a point onto the plane
  auto projectToPlane = [&](const pcl::PointXYZ &pt) -> geometry_msgs::msg::Point {
    geometry_msgs::msg::Point p;
    float t = -(best_plane[0] * pt.x + best_plane[1] * pt.y + best_plane[2] * pt.z
                + best_plane[3])
              / (best_plane[0] * best_plane[0] + best_plane[1] * best_plane[1]
                 + best_plane[2] * best_plane[2]);
    p.x = pt.x + best_plane[0] * t;
    p.y = pt.y + best_plane[1] * t;
    p.z = pt.z + best_plane[2] * t;
    return p;
  };

  // Get 4 corners of bounding box projected on the plane
  geometry_msgs::msg::Point p1 = projectToPlane({min_pt.x, min_pt.y, min_pt.z});
  geometry_msgs::msg::Point p2 = projectToPlane({max_pt.x, min_pt.y, min_pt.z});
  geometry_msgs::msg::Point p3 = projectToPlane({max_pt.x, max_pt.y, min_pt.z});
  geometry_msgs::msg::Point p4 = projectToPlane({min_pt.x, max_pt.y, min_pt.z});

  // Add two triangles for the plane surface
  plane_marker.points.push_back(p1);
  plane_marker.points.push_back(p2);
  plane_marker.points.push_back(p3);

  plane_marker.points.push_back(p1);
  plane_marker.points.push_back(p3);
  plane_marker.points.push_back(p4);
  marker_array.markers.push_back(plane_marker);
  plane_pub_->publish(marker_array);
}

void ClusterVision::publishPlane(
  const std::pair<pcl::ModelCoefficients, pcl::PointIndices> &best_plane,
  const pcl::PointCloud<pcl::PointXYZ> &cloud)
{
  const pcl::ModelCoefficients &coeff = best_plane.first;
  const pcl::PointIndices &indices = best_plane.second;

  // Extract plane points
  pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  for(int idx : indices.indices)
    plane_cloud->points.push_back(cloud.points[idx]);

  visualization_msgs::msg::MarkerArray marker_array;
  visualization_msgs::msg::Marker plane_marker;
  plane_marker.header.frame_id
    = base_frame_; // your fixed frame id, e.g. "map" or "base_link"
  plane_marker.header.stamp = this->now();
  plane_marker.ns = "detected_planes";
  plane_marker.id = 0; // unique id per marker
  plane_marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
  plane_marker.action = visualization_msgs::msg::Marker::ADD;
  plane_marker.scale.x = 1.0;
  plane_marker.scale.y = 1.0;
  plane_marker.scale.z = 1.0;
  plane_marker.color.r = 0.0f;
  plane_marker.color.g = 0.5f;
  plane_marker.color.b = 1.0f;
  plane_marker.color.a = 0.5f; // semi-transparent

  // Compute bounding box corners of the inliers
  pcl::PointXYZ min_pt, max_pt;
  pcl::getMinMax3D(*plane_cloud, min_pt, max_pt);

  // Lambda to project a point onto the plane
  auto projectToPlane = [&](const pcl::PointXYZ &pt) -> geometry_msgs::msg::Point {
    geometry_msgs::msg::Point p;
    float t = -(coeff.values[0] * pt.x + coeff.values[1] * pt.y + coeff.values[2] * pt.z
                + coeff.values[3])
              / (coeff.values[0] * coeff.values[0] + coeff.values[1] * coeff.values[1]
                 + coeff.values[2] * coeff.values[2]);
    p.x = pt.x + coeff.values[0] * t;
    p.y = pt.y + coeff.values[1] * t;
    p.z = pt.z + coeff.values[2] * t;
    return p;
  };

  // Get 4 corners of bounding box projected on the plane
  geometry_msgs::msg::Point p1 = projectToPlane({min_pt.x, min_pt.y, min_pt.z});
  geometry_msgs::msg::Point p2 = projectToPlane({max_pt.x, min_pt.y, min_pt.z});
  geometry_msgs::msg::Point p3 = projectToPlane({max_pt.x, max_pt.y, min_pt.z});
  geometry_msgs::msg::Point p4 = projectToPlane({min_pt.x, max_pt.y, min_pt.z});

  // Add two triangles for the plane surface
  plane_marker.points.push_back(p1);
  plane_marker.points.push_back(p2);
  plane_marker.points.push_back(p3);

  plane_marker.points.push_back(p1);
  plane_marker.points.push_back(p3);
  plane_marker.points.push_back(p4);

  marker_array.markers.push_back(plane_marker);
  plane_pub_->publish(marker_array);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ClusterVision>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
