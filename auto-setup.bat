@echo off
setlocal EnableDelayedExpansion

:: ---- Settings ----
set "GENERATOR=Visual Studio 17 2022"
set "ARCHITECTURE=x64"
set "BUILD_TYPE=Debug"

echo [Setup] Creating project folders...

:: ---- Create directories ----
if not exist src (
    mkdir src || (
        echo [Error] Failed to create src directory.
        exit /b 1
    )
)

if not exist vendor (
    mkdir vendor || (
        echo [Error] Failed to create vendor directory.
        exit /b 1
    )
)

:: ---- Download GLFW ----
if not exist vendor\glfw (
    echo [Clone] Cloning GLFW...
    git clone --depth 1 https://github.com/glfw/glfw vendor/glfw || (
        echo [Error] Failed to clone GLFW.
        exit /b 1
    )
)

:: ---- Download GLM ----
if not exist vendor\glm (
    echo [Clone] Cloning GLM...
    git clone --depth 1 https://github.com/g-truc/glm vendor/glm || (
        echo [Error] Failed to clone GLM.
        exit /b 1
    )
)

:: ---- Download ImGUI ----
if not exist vendor\imgui (
    echo [Clone] Cloning ImGUI...
    git clone --depth 1 https://github.com/ocornut/imgui vendor/imgui || (
        echo [Error] Failed to clone ImGUI.
        exit /b 1
    )
)

:: ---- Download Portaudio ----
if not exist vendor\portaudio (
    echo [Clone] Cloning PortAudio...
    git clone --depth 1 https://github.com/PortAudio/portaudio vendor/portaudio || (
        echo [Error] Failed to clone PortAudio.
        exit /b 1
    )
)

:: ---- Download fftw ----
if not exist vendor\fftw (
    echo [Clone] Cloning fftw...
    git clone --depth 1 https://github.com/PortAudio/portaudio vendor/fftw || (
        echo [Error] Failed to clone fftw.
        exit /b 1
    )
)


if not exist src\Main.cpp (
    echo [Generate] Writing Main.cpp...
    goto write_main
)
goto skip_main

:write_main
    >> src\Main.cpp echo #include ^<vulkan/vulkan.h^>
    >> src\Main.cpp echo #include ^<GLFW/glfw3.h^>
    >> src\Main.cpp echo #include ^<glm/glm.hpp^>
    >> src\Main.cpp echo #include ^<iostream^>
    >> src\Main.cpp echo.
    >> src\Main.cpp echo int main() {
    >> src\Main.cpp echo     glfwInit();
    >> src\Main.cpp echo     glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    >> src\Main.cpp echo     GLFWwindow* window = glfwCreateWindow(800, 600, "Vulkan Window", nullptr, nullptr);
    >> src\Main.cpp echo     while (!glfwWindowShouldClose(window)) {
    >> src\Main.cpp echo         glfwPollEvents();
    >> src\Main.cpp echo     }
    >> src\Main.cpp echo     glfwDestroyWindow(window);
    >> src\Main.cpp echo     glfwTerminate();
    >> src\Main.cpp echo     return 0;
    >> src\Main.cpp echo }

:skip_main

:: ---- Create CMakeLists.txt if it doesn't exist ----
if not exist CMakeLists.txt (
    echo [Generate] Writing CMakeLists.txt...
    goto write_cmake
)
goto skip_cmake

:write_cmake

    >> CMakeLists.txt echo cmake_minimum_required(VERSION 3.15)
    >> CMakeLists.txt echo project(VulkanApp)
    >> CMakeLists.txt echo set(CMAKE_CXX_STANDARD 20)
    >> CMakeLists.txt echo set(CMAKE_CXX_STANDARD_REQUIRED ON)
    >> CMakeLists.txt echo find_package(Vulkan REQUIRED)
    >> CMakeLists.txt echo add_subdirectory(vendor/glfw)
    >> CMakeLists.txt echo include_directories^(
    >> CMakeLists.txt echo     vendor/glfw/include
    >> CMakeLists.txt echo     vendor/glm
    >> CMakeLists.txt echo     ${Vulkan_INCLUDE_DIRS}
    >> CMakeLists.txt echo     ${CMAKE_SOURCE_DIR}/src
    >> CMakeLists.txt echo ^)
    >> CMakeLists.txt echo file(GLOB_RECURSE SOURCES "src/*.cpp")
    >> CMakeLists.txt echo add_executable(VulkanApp ${SOURCES})
    >> CMakeLists.txt echo target_link_libraries(VulkanApp Vulkan::Vulkan glfw)

if errorlevel 1 (
    echo [Error] Failed to create CMakeLists.txt.
    exit /b 1
)

:skip_cmake

:: ---- Create build directory ----
if not exist build (
    mkdir build || (
        echo [Error] Failed to create build directory.
        exit /b 1
    )
)

:: ---- Change to build directory ----
cd build || (
    echo [Error] Failed to change to build directory.
    exit /b 1
)

:: ---- Run CMake ----
echo [CMake] Configuring project...
cmake .. -G "%GENERATOR%" -A %ARCHITECTURE% -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
if errorlevel 1 (
    echo [Error] CMake configuration failed.
    exit /b 1
)

:: ---- Build Project ----
echo [Build] Building...
cmake --build . --config %BUILD_TYPE%
if errorlevel 1 (
    echo [Error] Build failed.
    exit /b 1
)

echo [Done] Vulkan app built successfully!

echo [Next Step] 
echo Open Visual Studio and choose the option to open an project in a existing folder. Pick the one with CMakeLists.txt file
echo Select the Main.cpp file and run

endlocal
pause