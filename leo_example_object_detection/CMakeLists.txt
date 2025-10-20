cmake_minimum_required(VERSION 3.5)
project(leo_example_object_detection)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

install(DIRECTORY
  config
  launch
  models
  DESTINATION share/${PROJECT_NAME}/
)

ament_python_install_package(leo_object_detection)

# Install Python executables
install(PROGRAMS
  scripts/object_detector
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()