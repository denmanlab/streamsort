# Non-JUCE test driver for the plugin's StreamDecoder/ZmqLink (macOS/Linux). Not part of the plugin build.
cmake_minimum_required(VERSION 3.15)
project(streamsort_plugin_path_test CXX)
set(CMAKE_CXX_STANDARD 17)
set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/../../../cpp/core)
set(ZMQ_OS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libs/macos)
add_executable(plugin_path_driver plugin_path_driver.cpp ../Source/StreamDecoder.cpp ../Source/ZmqLink.cpp
  ${CORE}/src/decoder.cpp ${CORE}/src/suppressor.cpp ${CORE}/src/bank.cpp)
target_include_directories(plugin_path_driver PRIVATE ${CORE}/include ${ZMQ_OS_DIR}/include)
target_compile_options(plugin_path_driver PRIVATE -O3 -ffp-contract=off)
find_library(ZMQ_LIB zmq PATHS ${ZMQ_OS_DIR}/lib NO_DEFAULT_PATH)
target_link_libraries(plugin_path_driver ${ZMQ_LIB} pthread)
set_target_properties(plugin_path_driver PROPERTIES BUILD_RPATH ${ZMQ_OS_DIR}/bin)
