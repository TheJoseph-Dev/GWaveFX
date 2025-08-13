cmake_minimum_required(VERSION 3.15)
project(GWaveFX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vulkan SDK
find_package(Vulkan REQUIRED)

# GLFW
add_subdirectory(vendor/glfw)
set(GLFW_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/vendor/glfw/include)

# GLM
set(GLM_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/vendor/glm)

# ImGui
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/vendor/imgui)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)
add_library(imgui STATIC ${IMGUI_SOURCES})

target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${GLFW_INCLUDE_DIR}
    ${Vulkan_INCLUDE_DIRS}
)

# PortAudio
set(PORTAUDIO_DIR ${CMAKE_SOURCE_DIR}/vendor/portaudio)
set(PA_BUILD_SHARED OFF CACHE BOOL "Don't build shared PortAudio")
set(PA_BUILD_STATIC ON CACHE BOOL "Build static PortAudio")
add_subdirectory(${PORTAUDIO_DIR} portaudio_build)

# FFTW
set(FFTW_DIR ${CMAKE_SOURCE_DIR}/vendor/fftw)

# Include directories
include_directories(
    ${GLFW_INCLUDE_DIR}
    ${GLM_INCLUDE_DIR}
    ${Vulkan_INCLUDE_DIRS}
    ${PORTAUDIO_DIR}/include
    ${FFTW_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.h")

foreach(source_file ${SOURCES})
    get_filename_component(source_path "${source_file}" PATH)
    file(RELATIVE_PATH source_path_rel "${CMAKE_SOURCE_DIR}/src" "${source_path}")
    string(REPLACE "/" "\\" source_group_name "${source_path_rel}")
    source_group("${source_group_name}" FILES "${source_file}")
endforeach()


add_executable(GWaveFX ${SOURCES})

# Link libraries
target_link_libraries(GWaveFX Vulkan::Vulkan glfw imgui portaudio_static ${FFTW_DIR}/libfftw3-3.lib)

# target_compile_definitions(GWaveFX PRIVATE FFTW_STATIC)
# Copy the fftw3-3.dll to .exe dir
add_custom_command(TARGET GWaveFX POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${FFTW_DIR}/libfftw3-3.dll"
    $<TARGET_FILE_DIR:GWaveFX>)