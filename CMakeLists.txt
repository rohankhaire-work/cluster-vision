cmake_minimum_required(VERSION 3.15)
project(cluster_vision CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VISUALIZE_FEATURES "Enable visualizing features" OFF)
option(GPU_ACCELERATION "Enable GPU acceleration for plane segmentation" ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PCL 1.2 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(cluster_vision src/cluster_vision_node.cpp
                              src/plane_segmentation_cuda.cu
                              src/plane_segmentation.cpp
                              )

if(VISUALIZE_FEATURES)
  target_compile_definitions(cluster_vision PRIVATE VISUALIZE_FEATURES)
endif()

if(GPU_ACCELERATION)
  target_compile_definitions(cluster_vision PRIVATE GPU_ACCELERATION) 
endif()

target_include_directories(cluster_vision PRIVATE include
                                            ${PCL_INCLUDE_DIRS}
                                            ${OpenCV_INCLUDE_DIRS})
                                            
ament_target_dependencies(cluster_vision rclcpp pcl_conversions pcl_ros tf2_ros tf2_geometry_msgs visualization_msgs)

target_link_libraries(cluster_vision
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen)

if(GPU_ACCELERATION)
  find_package(CUDAToolkit REQUIRED)
  target_include_directories(cluster_vision PRIVATE
    ${CUDAToolkit_INCLUDE_DIR}
  )
  target_link_libraries(cluster_vision
    ${CUDAToolkit_LIBRARY_DIR}/libcudart.so
  )
endif()

install(TARGETS cluster_vision
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
