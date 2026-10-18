# streamsort Open Ephys plugin (Plugin API v10, GUI v1.x). Based on the Open Ephys processor-plugin-template.
cmake_minimum_required(VERSION 3.15)
if (NOT DEFINED GUI_BASE_DIR)
  if (DEFINED ENV{GUI_BASE_DIR})
    set(GUI_BASE_DIR $ENV{GUI_BASE_DIR})
  else()
    set(GUI_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../plugin-GUI)
  endif()
endif()
# libzmq: headers + library (the Open Ephys zmq-interface repo ships builds in libs/<os>)
if (NOT DEFINED ZMQ_LIBS_DIR)
  set(ZMQ_LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs)
endif()
set(STREAMSORT_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/core)

set(PLUGIN_NAME streamsort-oe)
project(OE_PLUGIN_streamsort)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(LINUX 1)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS
  OEPLUGIN
  "$<$<PLATFORM_ID:Windows>:JUCE_API=__declspec(dllimport)>"
  $<$<PLATFORM_ID:Windows>:_CRT_SECURE_NO_WARNINGS>
  $<$<PLATFORM_ID:Linux>:JUCE_DISABLE_NATIVE_FILECHOOSERS=1>
  $<$<CONFIG:Debug>:DEBUG=1>
  $<$<CONFIG:Debug>:_DEBUG=1>
  $<$<CONFIG:Release>:NDEBUG=1>)

file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/Source/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Source/*.h)
set(CORE_FILES ${STREAMSORT_CORE_DIR}/src/decoder.cpp ${STREAMSORT_CORE_DIR}/src/suppressor.cpp ${STREAMSORT_CORE_DIR}/src/bank.cpp)
set_source_files_properties(${CORE_FILES} PROPERTIES COMPILE_OPTIONS "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3;-ffp-contract=off;-fno-fast-math>;$<$<CXX_COMPILER_ID:MSVC>:/O2;/fp:precise>")

set(GUI_COMMONLIB_DIR ${GUI_BASE_DIR}/installed_libs)
set(CONFIGURATION_FOLDER $<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>)
if (APPLE)
  add_library(${PLUGIN_NAME} MODULE ${SRC_FILES} ${CORE_FILES})
else()
  add_library(${PLUGIN_NAME} SHARED ${SRC_FILES} ${CORE_FILES})
endif()
target_compile_features(${PLUGIN_NAME} PUBLIC cxx_std_17)
target_include_directories(${PLUGIN_NAME} PUBLIC ${GUI_BASE_DIR}/JuceLibraryCode ${GUI_BASE_DIR}/JuceLibraryCode/modules
  ${GUI_BASE_DIR}/Plugins/Headers ${GUI_COMMONLIB_DIR}/include ${STREAMSORT_CORE_DIR}/include)
set(GUI_BIN_DIR ${GUI_BASE_DIR}/Build/${CONFIGURATION_FOLDER})
if (NOT CMAKE_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "x64")
endif()

if(MSVC)
  target_link_libraries(${PLUGIN_NAME} ${GUI_BIN_DIR}/open-ephys.lib)
  target_compile_options(${PLUGIN_NAME} PRIVATE /sdl- /W0)
  install(TARGETS ${PLUGIN_NAME} RUNTIME DESTINATION ${GUI_BIN_DIR}/plugins CONFIGURATIONS ${CMAKE_CONFIGURATION_TYPES})
  set(ZMQ_OS_DIR ${ZMQ_LIBS_DIR}/windows)
  install(DIRECTORY ${ZMQ_OS_DIR}/bin/${CMAKE_LIBRARY_ARCHITECTURE}/ DESTINATION ${GUI_BIN_DIR}/shared CONFIGURATIONS ${CMAKE_CONFIGURATION_TYPES})
elseif(LINUX)
  target_link_libraries(${PLUGIN_NAME} GL X11 Xext Xinerama asound dl freetype pthread rt)
  set_property(TARGET ${PLUGIN_NAME} APPEND_STRING PROPERTY LINK_FLAGS
    "-fvisibility=hidden -fPIC -rdynamic -Wl,-rpath='$ORIGIN/../shared' -Wl,-rpath='$ORIGIN/../shared-api10'")
  target_compile_options(${PLUGIN_NAME} PRIVATE -fPIC -rdynamic -O3)
  install(TARGETS ${PLUGIN_NAME} LIBRARY DESTINATION ${GUI_BIN_DIR}/plugins)
  set(ZMQ_OS_DIR ${ZMQ_LIBS_DIR}/linux)
  install(DIRECTORY ${ZMQ_OS_DIR}/bin/ DESTINATION ${GUI_BIN_DIR}/shared)
elseif(APPLE)
  set_target_properties(${PLUGIN_NAME} PROPERTIES BUNDLE TRUE)
  set_property(TARGET ${PLUGIN_NAME} APPEND_STRING PROPERTY LINK_FLAGS
    "-undefined dynamic_lookup -rpath @loader_path/../../../../shared-api10")
  install(TARGETS ${PLUGIN_NAME} DESTINATION $ENV{HOME}/Library/Application\ Support/open-ephys/plugins-api10)
  set(ZMQ_OS_DIR ${ZMQ_LIBS_DIR}/macos)
  install(DIRECTORY ${ZMQ_OS_DIR}/bin/ DESTINATION $ENV{HOME}/Library/Application\ Support/open-ephys/shared-api10)
endif()

find_library(ZMQ_LIBRARIES NAMES libzmq-v142-mt-4_3_4 zmq zmq-v142-mt-4_3_4 PATHS ${ZMQ_OS_DIR}/lib NO_DEFAULT_PATH)
find_path(ZMQ_INCLUDE_DIRS zmq.h PATHS ${ZMQ_OS_DIR}/include NO_DEFAULT_PATH)
if (NOT ZMQ_LIBRARIES OR NOT ZMQ_INCLUDE_DIRS)
  message(FATAL_ERROR "libzmq not found under ${ZMQ_OS_DIR} (copy libs/ from open-ephys-plugins/zmq-interface or set ZMQ_LIBS_DIR)")
endif()
target_include_directories(${PLUGIN_NAME} PUBLIC ${ZMQ_INCLUDE_DIRS})
target_link_libraries(${PLUGIN_NAME} ${ZMQ_LIBRARIES})
