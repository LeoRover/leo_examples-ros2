cmake_minimum_required(VERSION 3.5)
project(leo_example_line_follower)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(generate_parameter_library REQUIRED)

generate_parameter_module(follower_parameters
  leo_line_follower/follower_parameters.yaml
)

install(DIRECTORY
  config
  launch
  models
  DESTINATION share/${PROJECT_NAME}/
)

ament_python_install_package(leo_line_follower)

# Install Python executables
install(PROGRAMS
  scripts/color_mask
  scripts/data_saver
  scripts/line_follower
  scripts/prepare_data
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()