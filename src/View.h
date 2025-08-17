#include "Vulkan.h"

namespace GWaveFX {

    class View {

        GLFWwindow* window;
        Vulkan vulkan;

    public:
        View() {
            initWindow();
            vulkan.init(window);
        }

        ~View() {
            vulkan.shutdown();
        }

        void run() {
            mainLoop();
        }

    private:
        void initWindow() {
            glfwInit();
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
            this->window = glfwCreateWindow(GWaveFX::WIDTH, GWaveFX::HEIGHT, "GWaveFX", nullptr, nullptr);
        }

        void mainLoop() {

            uint32_t currentFrame = 0;

            while (!glfwWindowShouldClose(window)) {
                vulkan.loop(currentFrame);
                glfwPollEvents();
            }
        }

    };
}