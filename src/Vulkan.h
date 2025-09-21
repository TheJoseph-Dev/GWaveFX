#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <set>
#include <fstream>
#include <direct.h>
#include <optional>
#include <glm/glm.hpp>

namespace GWaveFX {
    constexpr uint32_t WIDTH = 1080;
    constexpr uint32_t HEIGHT = 720;
}

// How many frames you let the CPU start rendering before the GPU is done
// MAX_FRAMES_IN_FLIGHT: How many frames the CPU can work on at once
// Swap Chain: How many images are available to render into
// They are not directly tied, but you need to consider things like: MAX_FRAMES_IN_FLIGHT = min(MAX_FRAMES_IN_FLIGHT, SC.size())
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

std::string RESOURCE_PATH_PREFIX = std::string("..\\..\\..\\src\\");
#define RESOURCE(filepath) "..\\..\\..\\src\\" filepath

#define DEBUG
#ifdef DEBUG
constexpr bool enableValidationLayers = true;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};
#else
constexpr bool enableValidationLayers = false;
#endif

struct Vertex {
    glm::vec2 pos;
    glm::vec2 uv;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static void getAttributeDescriptions(VkVertexInputAttributeDescription attributeDescriptions[2]) {
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, uv);
    }
};

class Vulkan {
    
    VkInstance instance; // Connection between Vulkan and the main program
    VkPhysicalDevice physicalDevice;
    VkDevice device; // After selecting a Physical Device, create a logical device to interface with it
    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    uint32_t graphicsQueueFamilyIndex = 0, presentQueueFamilyIndex = 0, computeQueueFamilyIndex = 0;
    VkQueue graphicsQueue; // Handle to interact with device graphics queue;
    VkSurfaceKHR surface; // Handle to interact with window
    VkQueue presentQueue; // Handle to interact with window surface queue;

    VkSurfaceFormatKHR surfaceFormat;
    VkExtent2D extent;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;

    VkViewport viewport;
    VkRect2D scissor; // Cut viewport filter >:/

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet fragDescriptorSet;

    VkImage dataImage[2];
    VkDeviceMemory dataImageMemory[2];
    VkImageView dataTexture[2];
    VkSampler dataSampler;

    VkBuffer vertexBuffer;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkCommandPool commandPool;
    VkCommandBuffer commandBuffers[MAX_FRAMES_IN_FLIGHT];

    VkSemaphore imageAvailableSemaphores[MAX_FRAMES_IN_FLIGHT];
    VkSemaphore renderFinishedSemaphores[MAX_FRAMES_IN_FLIGHT];
    VkFence inFlightFences[MAX_FRAMES_IN_FLIGHT];

public:
    Vulkan() {}

    void init(GLFWwindow* window) {
        createInstance();
        createWindowSurface(window);
        createDevice();
        createSwapchain(window);
        createGraphicsPipeline();
    }

    void createInstance() {
        std::cout << "\n Vulkan Header Version: " << VK_HEADER_VERSION << std::endl;
        std::cout << " Vulkan API Version: " << VK_API_VERSION_VARIANT(VK_HEADER_VERSION_COMPLETE)
            << "." << VK_API_VERSION_MAJOR(VK_HEADER_VERSION_COMPLETE)
            << "." << VK_API_VERSION_MINOR(VK_HEADER_VERSION_COMPLETE)
            << std::endl;

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "GWaveFX";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo instanceInfo{};
        instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceInfo.pApplicationInfo = &appInfo;

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        instanceInfo.enabledExtensionCount = glfwExtensionCount;
        instanceInfo.ppEnabledExtensionNames = glfwExtensions;
        instanceInfo.enabledLayerCount = 0;

#ifdef DEBUG
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        bool layersAvailable = true;
        for (const char* layerName : validationLayers) {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers)
                if (strcmp(layerName, layerProperties.layerName) == 0) { layerFound = true; break; }
            if (!layerFound) std::cerr << "Requested validation layer: " << layerName << " is not availible\n";
            layersAvailable &= layerFound;
        }

        if (layersAvailable) {
            instanceInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            instanceInfo.ppEnabledLayerNames = validationLayers.data();
        }
#endif

        if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
            std::cerr << "VkInstance Creation Error\n";
            return;
        }
    }

    void createWindowSurface(GLFWwindow* window) {
        if (glfwCreateWindowSurface(this->instance, window, nullptr, &this->surface) != VK_SUCCESS) {
            std::cerr << "VkSurfaceKHR Creation Error\n";
            return;
        }
    }

    void createDevice() {
        this->physicalDevice = VK_NULL_HANDLE;

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (!deviceCount) { std::cerr << "Failed to find GPUs with Vulkan support\n"; return; }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        physicalDevice = devices[0];

        std::cout << "\n Available Devices:\n";
        VkPhysicalDeviceProperties deviceProperties;
        VkPhysicalDeviceFeatures2 deviceFeatures2;
        VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{};
        dynamicRenderingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;

        for (const auto& device : devices) {
            vkGetPhysicalDeviceProperties(device, &deviceProperties);

            deviceFeatures2 = {};
            deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            deviceFeatures2.pNext = &dynamicRenderingFeatures;
            vkGetPhysicalDeviceFeatures2(device, &deviceFeatures2);
            std::cout << "\n - " << deviceProperties.deviceName;

            if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                std::cout << " (Discrete)";
                physicalDevice = device;
            }

            if (dynamicRenderingFeatures.dynamicRendering) {
                std::cout << " (Core Dyn. Rendering)";
            }

            uint32_t extensionCount;
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

            std::vector<VkExtensionProperties> availableExtensions(extensionCount);
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

            std::set<std::string> requiredExtensions(this->deviceExtensions.begin(), this->deviceExtensions.end());
            for (const auto& extension : availableExtensions) requiredExtensions.erase(extension.extensionName);

            if (requiredExtensions.empty()) std::cout << " (Extensions Available)";

            uint32_t apiVersion = deviceProperties.apiVersion;
            uint32_t major = VK_VERSION_MAJOR(apiVersion);
            uint32_t minor = VK_VERSION_MINOR(apiVersion);
            uint32_t patch = VK_VERSION_PATCH(apiVersion);

            std::cout << " (" << major << "." << minor << "." << patch << ")";

            std::cout << std::endl;
        }
        std::cout << std::endl;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

        
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
        for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                graphicsQueueFamilyIndex = i;
                std::cout << " Queue family " << i << " supports graphics operations\n";
            };
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                computeQueueFamilyIndex = i;
                std::cout << " Queue family " << i << " supports compute operations\n";
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, this->surface, &presentSupport);
            if (presentSupport) {
                presentQueueFamilyIndex = i;
                std::cout << " Queue family " << i << " supports window surface\n";
            }
        }


        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> queueFamilyIndicies = { graphicsQueueFamilyIndex, presentQueueFamilyIndex, computeQueueFamilyIndex };
        for (uint32_t queueFamilyIndex : queueFamilyIndicies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
            queueCreateInfo.queueCount = 1;
            float queuePriority = 1.0f;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        dynamicRenderingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
        dynamicRenderingFeatures.dynamicRendering = VK_TRUE;

        deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        deviceFeatures2.features = {};
        deviceFeatures2.pNext = &dynamicRenderingFeatures;

        VkDeviceCreateInfo deviceInfo{};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.pNext = &dynamicRenderingFeatures;
        deviceInfo.pQueueCreateInfos = queueCreateInfos.data();
        deviceInfo.queueCreateInfoCount = queueCreateInfos.size();

        deviceInfo.pEnabledFeatures = nullptr;
        deviceInfo.pNext = &deviceFeatures2;
        deviceInfo.enabledExtensionCount = this->deviceExtensions.size();
        deviceInfo.ppEnabledExtensionNames = this->deviceExtensions.data();

        if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &this->device) != VK_SUCCESS) {
            std::cerr << "VkDevice Creation Error\n";
            return;
        }
    }

    void createSwapchain(GLFWwindow* window) {
        vkGetDeviceQueue(this->device, graphicsQueueFamilyIndex, 0, &this->graphicsQueue); // 0 because we created only 1 queue of this family
        vkGetDeviceQueue(this->device, presentQueueFamilyIndex, 0, &this->presentQueue);

        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, this->surface, &capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, this->surface, &formatCount, nullptr);

        if (formatCount) {
            formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, this->surface, &formatCount, formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, this->surface, &presentModeCount, nullptr);

        if (presentModeCount) {
            presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, this->surface, &presentModeCount, presentModes.data());
        }

        if (!formatCount || !presentModeCount) {
            std::cerr << "Chosen physical device swap chain doesn't support current window surface or is not adequate\n";
            return;
        }

        VkPresentModeKHR presentMode = presentModes[0];
        this->surfaceFormat = formats[0];
        this->extent = capabilities.currentExtent;

        for (const auto& sFormat : formats)
            if (sFormat.format == VK_FORMAT_B8G8R8A8_SRGB && sFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                this->surfaceFormat = sFormat;
                break;
            }

        for (const auto& pMode : presentModes)
            if (pMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                presentMode = pMode;
                break;
            }

        if (capabilities.currentExtent.width == std::numeric_limits<uint32_t>::max()) {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
            this->extent = actualExtent;
        }

        uint32_t imageCount = MAX_FRAMES_IN_FLIGHT; // How many images to have in the swap chain
        //(capabilities.maxImageCount > 0 && capabilities.minImageCount + 1 > capabilities.maxImageCount) ?
        //capabilities.maxImageCount : capabilities.minImageCount + 1; 

        VkSwapchainCreateInfoKHR swapChainInfo{};
        swapChainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        swapChainInfo.surface = this->surface;
        swapChainInfo.minImageCount = imageCount;
        swapChainInfo.imageFormat = this->surfaceFormat.format;
        swapChainInfo.imageColorSpace = this->surfaceFormat.colorSpace;
        swapChainInfo.imageExtent = this->extent;
        swapChainInfo.imageArrayLayers = 1;
        swapChainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        swapChainInfo.preTransform = capabilities.currentTransform;
        swapChainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        swapChainInfo.presentMode = presentMode;
        swapChainInfo.clipped = VK_TRUE;
        swapChainInfo.oldSwapchain = VK_NULL_HANDLE;

        if (graphicsQueueFamilyIndex != presentQueueFamilyIndex) {
            swapChainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            uint32_t queueFamilyIndices[] = { graphicsQueueFamilyIndex, presentQueueFamilyIndex };
            swapChainInfo.queueFamilyIndexCount = sizeof(queueFamilyIndices) / sizeof(uint32_t);
            swapChainInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            swapChainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            swapChainInfo.queueFamilyIndexCount = 0;
            swapChainInfo.pQueueFamilyIndices = nullptr;
        }

        if (vkCreateSwapchainKHR(this->device, &swapChainInfo, nullptr, &this->swapChain) != VK_SUCCESS) {
            std::cerr << "VkSwapchainKHR Creation Error\n";
            return;
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageViews.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo imageViewInfo{};
            imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            imageViewInfo.image = swapChainImages[i];
            imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            imageViewInfo.format = swapChainInfo.imageFormat;
            imageViewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            imageViewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            imageViewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            imageViewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            imageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageViewInfo.subresourceRange.baseMipLevel = 0;
            imageViewInfo.subresourceRange.levelCount = 1;
            imageViewInfo.subresourceRange.baseArrayLayer = 0;
            imageViewInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(this->device, &imageViewInfo, nullptr, &this->swapChainImageViews[i]) != VK_SUCCESS) {
                std::cerr << "VkImageView Creation Error\n";
                return;
            }
        }
    }

    enum ShaderType {
        VERTEX,
        FRAGMENT,
        COMPUTE
    };

    std::optional<VkShaderModule> createShaderModule(const std::string& path, ShaderType type) {
        std::ifstream shaderFile(RESOURCE_PATH_PREFIX + path, std::ios::ate | std::ios::binary);
        if (!shaderFile.is_open()) { std::cerr << "Failed to read shader file\n"; return std::nullopt; }

        size_t shaderFileSize = static_cast<size_t>(shaderFile.tellg());
        std::vector<char> shaderBuffer(shaderFileSize);

        shaderFile.seekg(0);
        shaderFile.read(shaderBuffer.data(), shaderFileSize);
        shaderFile.close();

        VkShaderModule shaderModule;
        VkShaderModuleCreateInfo shaderModuleInfo{};
        shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderModuleInfo.codeSize = shaderBuffer.size();
        shaderModuleInfo.pCode = reinterpret_cast<const uint32_t*>(shaderBuffer.data());
        if (vkCreateShaderModule(this->device, &shaderModuleInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            std::cerr << "Failed to create VkShaderModule\n"; 
            return std::nullopt;
        }

        return shaderModule;
    }

    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = this->commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(this->device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(this->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(this->graphicsQueue);

        vkFreeCommandBuffers(this->device, this->commandPool, 1, &commandBuffer);
    }

    void createDescriptorSetLayout(const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        vkCreateDescriptorSetLayout(this->device, &layoutInfo, nullptr, &this->descriptorSetLayout);
    }

    void createDescriptorPool(const std::vector<VkDescriptorPoolSize>& poolSizes, uint32_t maxSets) {
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = maxSets;

        vkCreateDescriptorPool(this->device, &poolInfo, nullptr, &this->descriptorPool);
    }

    VkDescriptorSet allocateDescriptorSet() {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = this->descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &this->descriptorSetLayout;

        VkDescriptorSet descriptorSet;
        vkAllocateDescriptorSets(this->device, &allocInfo, &descriptorSet);
        return descriptorSet;
    }

    void updateUniformBufferDescriptorSet(VkDescriptorSet descriptorSet, uint32_t binding, VkBuffer buffer, VkDeviceSize size) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = size;

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descriptorSet;
        descriptorWrite.dstBinding = binding;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(this->device, 1, &descriptorWrite, 0, nullptr);
    }

    struct UniformBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    };

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(this->device, buffer, bufferMemory, 0);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;

        throw std::runtime_error("failed to find suitable memory type!");
    }

    UniformBuffer createUniformBuffer(VkDeviceSize size, const void* initialData) {
        UniformBuffer ubo;

        createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, ubo.buffer, ubo.memory);

        // Upload initial data
        if (initialData) {
            void* data;
            vkMapMemory(device, ubo.memory, 0, size, 0, &data);
            memcpy(data, initialData, (size_t)size);
            vkUnmapMemory(device, ubo.memory);
        }

        return ubo;
    }

    /*
        | Scenario                             | Use Staging Buffer? | Why                                                                                    |
        | ------------------------------------ | ------------------- | -------------------------------------------------------------------------------------- |
        | Discrete GPU, infrequent UBO updates | Yes                 | Best GPU performance; device-local memory is fast                                      |
        | Integrated GPU (shared memory)       | No                  | No real benefit from copying; adds complexity                                          |
        | Frequent updates per frame           | Maybe not           | Copying becomes a bottleneck; better to use ring buffers or persistently mapped memory |
        | Need maximum GPU performance         | Yes                 | Allows GPU-optimal memory usage                                                        |
    
        Therefore, when having to update an image many times it is better to not stage at all
    */

    void createImage1D(uint32_t size, VkDescriptorSet descriptorSet) {
        //VkImage dataImage; // Gotta destroy it later
        //VkImageView dataTexture;
        //VkSampler dataSampler;
        
        VkSamplerCreateInfo dataSamplerInfo{};
        dataSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        dataSamplerInfo.magFilter = VK_FILTER_LINEAR;
        dataSamplerInfo.minFilter = VK_FILTER_LINEAR;
        dataSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        dataSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        dataSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        dataSamplerInfo.anisotropyEnable = VK_FALSE;
        dataSamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        dataSamplerInfo.unnormalizedCoordinates = VK_FALSE;

        vkCreateSampler(this->device, &dataSamplerInfo, nullptr, &dataSampler);

        // Double Buffer
        for (int i = 0; i < 2; i++) {
            VkImageCreateInfo dataImageInfo{};
            dataImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            dataImageInfo.imageType = VK_IMAGE_TYPE_1D;
            dataImageInfo.extent.width = size;
            dataImageInfo.extent.height = 1;
            dataImageInfo.extent.depth = 1;
            dataImageInfo.mipLevels = 1;
            dataImageInfo.arrayLayers = 1;
            dataImageInfo.format = VK_FORMAT_R32_SFLOAT;
            dataImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            dataImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            dataImageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            dataImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            dataImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

            vkCreateImage(this->device, &dataImageInfo, nullptr, &this->dataImage[i]);

            VkMemoryRequirements memRequirements;
            vkGetImageMemoryRequirements(this->device, this->dataImage[i], &memRequirements);

            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

            if (vkAllocateMemory(this->device, &allocInfo, nullptr, &this->dataImageMemory[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate image memory!");
            }

            vkBindImageMemory(this->device, dataImage[i], dataImageMemory[i], 0);

            VkImageViewCreateInfo dataTextureInfo{};
            dataTextureInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            dataTextureInfo.image = this->dataImage[i];
            dataTextureInfo.viewType = VK_IMAGE_VIEW_TYPE_1D;
            dataTextureInfo.format = VK_FORMAT_R32_SFLOAT;
            dataTextureInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            dataTextureInfo.subresourceRange.baseMipLevel = 0;
            dataTextureInfo.subresourceRange.levelCount = 1;
            dataTextureInfo.subresourceRange.baseArrayLayer = 0;
            dataTextureInfo.subresourceRange.layerCount = 1;

            vkCreateImageView(this->device, &dataTextureInfo, nullptr, &dataTexture[i]);

            VkDescriptorSetLayoutBinding imageSamplerBinding{};
            imageSamplerBinding.binding = 1;
            imageSamplerBinding.descriptorCount = 1;
            imageSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            imageSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            imageSamplerBinding.pImmutableSamplers = nullptr;

            VkDescriptorImageInfo descriptorImageInfo{};
            descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            descriptorImageInfo.imageView = dataTexture[i];
            descriptorImageInfo.sampler = dataSampler;

            VkWriteDescriptorSet samplerWrite{};
            samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            samplerWrite.dstSet = descriptorSet;
            samplerWrite.dstBinding = 1;
            samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            samplerWrite.pImageInfo = &descriptorImageInfo;
            samplerWrite.descriptorCount = 1;

            vkUpdateDescriptorSets(this->device, 1, &samplerWrite, 0, nullptr);
        }
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        /*
            One of the most common ways to perform layout transitions is using an image memory barrier. A pipeline barrier like that is generally
            used to synchronize access to resources, like ensuring that a write to a buffer completes before reading from it,
            but it can also be used to transition image layouts and transfer queue family ownership when VK_SHARING_MODE_EXCLUSIVE is used.
            There is an equivalent buffer memory barrier to do this for buffers.
        */

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage, destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage1D(VkBuffer buffer, VkImage image, uint32_t size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = {
            size,
            1,
            1
        };

        vkCmdCopyBufferToImage(
            commandBuffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );

        endSingleTimeCommands(commandBuffer);
    }

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    void* mappedStagedData;
    int textureIndex = 0;
    void createUniformBufferTexture1D(const float* dataBuffer, uint32_t size, VkDescriptorSet descriptorSet) {
        createBuffer(size * sizeof(float), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        vkMapMemory(this->device, stagingBufferMemory, 0, size * sizeof(float), 0, &mappedStagedData);
        memcpy(this->mappedStagedData, dataBuffer, static_cast<size_t>(size*sizeof(float)));
        //vkUnmapMemory(this->device, stagingBufferMemory);

        createImage1D(size, descriptorSet);
        //transitionImageLayout(this->dataImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        //copyBufferToImage1D(stagingBuffer, this->dataImage, static_cast<uint32_t>(size));
        //transitionImageLayout(this->dataImage[i], VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        //vkDestroyBuffer(this->device, stagingBuffer, nullptr);
        //vkFreeMemory(this->device, stagingBufferMemory, nullptr);
    }
    
    void createGraphicsPipeline() {
        // Graphics Pipeline
        VkCommandPoolCreateInfo cmdPoolInfo{};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cmdPoolInfo.queueFamilyIndex = graphicsQueueFamilyIndex;

        if (vkCreateCommandPool(this->device, &cmdPoolInfo, nullptr, &this->commandPool) != VK_SUCCESS) {
            std::cerr << "Failed to create VkCreateCommandPool\n";
            return;
        }

        // Gotta change cmake.txt
        // char buffer[FILENAME_MAX];
        // _getcwd(buffer, FILENAME_MAX);
        // std::cout << "Current working directory: " << buffer << std::endl;

        // Shaders
        system(RESOURCE("Shaders\\runtime_compile.bat"));
        VkShaderModule vsShaderModule = this->createShaderModule("Shaders\\vert.spv", ShaderType::VERTEX).value();
        VkShaderModule fsShaderModule = this->createShaderModule("Shaders\\frag.spv", ShaderType::FRAGMENT).value();

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vsShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fsShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        glm::vec4 resolution = glm::vec4(GWaveFX::WIDTH, GWaveFX::HEIGHT, 0.0f, 0.0f);
        UniformBuffer resolutionUBO = createUniformBuffer(sizeof(glm::vec4), &resolution);
        std::vector<VkDescriptorSetLayoutBinding> bindings = {
            { 0,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }
        };

        createDescriptorSetLayout(bindings);

        std::vector<VkDescriptorPoolSize> poolSizes = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }
        };

        createDescriptorPool(poolSizes, 1);

        this->fragDescriptorSet = allocateDescriptorSet();
        updateUniformBufferDescriptorSet(this->fragDescriptorSet, 0, resolutionUBO.buffer, sizeof(glm::vec4));

        const uint32_t bufferSize = 512;
        float dataBuffer[bufferSize] = { 0 };
        for (int i = 0; i < bufferSize; ++i) dataBuffer[i] = sin(i * 0.0174532925 * 10);
        createUniformBufferTexture1D(dataBuffer, bufferSize, this->fragDescriptorSet);

        // Dynamic State
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
        dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicStateInfo.pDynamicStates = dynamicStates.data();

        // Vertex Buffer
        std::vector<Vertex> vertices = {
            {{-1.0f, -1.0f}, {0.0f, 0.0f}},
            {{ 1.0f, -1.0f}, {1.0f, 0.0f}},
            {{-1.0f,  1.0f}, {0.0f, 1.0f}},

            {{-1.0f,  1.0f}, {0.0f, 1.0f}},
            {{ 1.0f, -1.0f}, {1.0f, 0.0f}},
            {{ 1.0f,  1.0f}, {1.0f, 1.0f}}
        };

        VkDeviceMemory vertexBufferMemory;
        VkBufferCreateInfo vertexBufferInfo{};
        vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        vertexBufferInfo.size = sizeof(vertices[0]) * vertices.size();
        vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        vertexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(this->device, &vertexBufferInfo, nullptr, &this->vertexBuffer) != VK_SUCCESS) {
            std::cerr << "Failed to create vertex buffer\n";
            return;
        }

        VkMemoryRequirements vertexBufferMemRequirements;
        vkGetBufferMemoryRequirements(this->device, this->vertexBuffer, &vertexBufferMemRequirements);

        VkPhysicalDeviceMemoryProperties vertexBufferMemProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &vertexBufferMemProperties);
        int memType = -1;
        for (uint32_t i = 0; i < vertexBufferMemProperties.memoryTypeCount && memType == -1; i++)
            if ((vertexBufferMemRequirements.memoryTypeBits & (1 << i)) &&
                (vertexBufferMemProperties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                memType = i;

        VkMemoryAllocateInfo vertexBufferAllocInfo{};
        vertexBufferAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        vertexBufferAllocInfo.allocationSize = vertexBufferMemRequirements.size;
        vertexBufferAllocInfo.memoryTypeIndex = memType;

        if (vkAllocateMemory(this->device, &vertexBufferAllocInfo, nullptr, &vertexBufferMemory) != VK_SUCCESS) {
            std::cerr << "Failed to allocate vertex buffer\n";
            return;
        }

        vkBindBufferMemory(this->device, this->vertexBuffer, vertexBufferMemory, 0);

        // Once the vertex buffer is allocated and created, it's time to fill it with data
        void* data;
        vkMapMemory(this->device, vertexBufferMemory, 0, vertexBufferInfo.size, 0, &data);
        memcpy(data, vertices.data(), vertexBufferInfo.size);
        vkUnmapMemory(this->device, vertexBufferMemory);

        /*
            Unfortunately the driver may not immediately copy the data into the buffer memory, for example because of caching.
            It is also possible that writes to the buffer are not visible in the mapped memory yet.
            There are two ways to deal with that problem:

                Use a memory heap that is host coherent, indicated with VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                Call vkFlushMappedMemoryRanges after writing to the mapped memory, and call vkInvalidateMappedMemoryRanges before reading from the mapped memory

            We went for the first approach, which ensures that the mapped memory always matches the contents
            of the allocated memory. Do keep in mind that this may lead to slightly worse performance than explicit
            flushing.
            For a improved performance, use staging buffers.
        */

        // Vertex Input Layout
        auto bindingDescription = Vertex::getBindingDescription();
        VkVertexInputAttributeDescription attributeDescriptions[2];
        Vertex::getAttributeDescriptions(attributeDescriptions);

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = 2;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo{};
        inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssemblyInfo.primitiveRestartEnable = VK_FALSE;


        this->viewport.x = 0.0f;
        this->viewport.y = 0.0f;
        this->viewport.width = (float)this->extent.width;
        this->viewport.height = (float)this->extent.height;
        this->viewport.minDepth = 0.0f;
        this->viewport.maxDepth = 1.0f;

        this->scissor.offset = { 0, 0 };
        this->scissor.extent = this->extent;

        VkPipelineViewportStateCreateInfo viewportStateInfo{};
        viewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportStateInfo.viewportCount = 1;
        viewportStateInfo.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizerInfo{};
        rasterizerInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizerInfo.depthClampEnable = VK_FALSE;
        rasterizerInfo.rasterizerDiscardEnable = VK_FALSE;
        rasterizerInfo.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizerInfo.lineWidth = 1.0f;
        rasterizerInfo.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizerInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizerInfo.depthBiasEnable = VK_FALSE;
        rasterizerInfo.depthBiasConstantFactor = 0.0f;
        rasterizerInfo.depthBiasClamp = 0.0f;
        rasterizerInfo.depthBiasSlopeFactor = 0.0f;


        VkPipelineMultisampleStateCreateInfo multisamplingInfo{};
        multisamplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisamplingInfo.sampleShadingEnable = VK_FALSE;
        multisamplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisamplingInfo.minSampleShading = 1.0f;
        multisamplingInfo.pSampleMask = nullptr;
        multisamplingInfo.alphaToCoverageEnable = VK_FALSE;
        multisamplingInfo.alphaToOneEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE; // VK_TRUE
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // VK_BLEND_FACTOR_SRC_ALPHA
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlendingInfo{}; // Array of structures for all of the framebuffers
        colorBlendingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendingInfo.logicOpEnable = VK_FALSE;
        colorBlendingInfo.logicOp = VK_LOGIC_OP_COPY;
        colorBlendingInfo.attachmentCount = 1;
        colorBlendingInfo.pAttachments = &colorBlendAttachment;
        colorBlendingInfo.blendConstants[0] = 0.0f;
        colorBlendingInfo.blendConstants[1] = 0.0f;
        colorBlendingInfo.blendConstants[2] = 0.0f;
        colorBlendingInfo.blendConstants[3] = 0.0f;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &this->descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &this->pipelineLayout) != VK_SUCCESS) {
            std::cerr << "Failed to create VkCreatePipelineLayout\n";
            return;
        }

        VkPipelineRenderingCreateInfo pipelineRenderingInfo{};
        pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        pipelineRenderingInfo.colorAttachmentCount = 1;
        pipelineRenderingInfo.pColorAttachmentFormats = &this->surfaceFormat.format;

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.pNext = &pipelineRenderingInfo; // this is essential for dynamic rendering!
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
        pipelineInfo.pViewportState = &viewportStateInfo;
        pipelineInfo.pRasterizationState = &rasterizerInfo;
        pipelineInfo.pMultisampleState = &multisamplingInfo;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlendingInfo;
        pipelineInfo.pDynamicState = &dynamicStateInfo;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = VK_NULL_HANDLE;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = -1;

        if (vkCreateGraphicsPipelines(this->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &this->graphicsPipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create VkCreatePipeline\n";
            return;
        }

        vkDestroyShaderModule(device, vsShaderModule, nullptr);
        vkDestroyShaderModule(device, fsShaderModule, nullptr);

        // 
        VkCommandBufferAllocateInfo cmdBufferAllocInfo{};
        cmdBufferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdBufferAllocInfo.commandPool = this->commandPool;
        cmdBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdBufferAllocInfo.commandBufferCount = (uint32_t)MAX_FRAMES_IN_FLIGHT;

        if (vkAllocateCommandBuffers(this->device, &cmdBufferAllocInfo, this->commandBuffers) != VK_SUCCESS) {
            std::cerr << "Failed to allocate with VkAllocateCommandBuffers\n";
            return;
        }

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // Workaround to avoid vkWaitForFences block the first frame forever

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            if (vkCreateSemaphore(this->device, &semaphoreInfo, nullptr, this->imageAvailableSemaphores + i) != VK_SUCCESS ||
                vkCreateSemaphore(this->device, &semaphoreInfo, nullptr, this->renderFinishedSemaphores + i) != VK_SUCCESS ||
                vkCreateFence(this->device, &fenceInfo, nullptr, this->inFlightFences + i) != VK_SUCCESS) {
                std::cerr << "Failed to create a VkSemaphore or VkFence\n";
                return;
            }
        }
    }

    void shutdown() {
        vkDeviceWaitIdle(this->device); // Ensures proper cleanup

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        for (auto imageView : swapChainImageViews)
            vkDestroyImageView(device, imageView, nullptr);

        vkDestroySwapchainKHR(device, swapChain, nullptr);

        for(int i = 0; i < 2; i++)
            vkDestroyImage(device, dataImage[i], nullptr);
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
    }

    void loop(uint32_t& currentFrame) {
        vkWaitForFences(device, 1, this->inFlightFences + currentFrame, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, this->inFlightFences + currentFrame);

        uint32_t imageIndex;
        vkAcquireNextImageKHR(this->device, this->swapChain, UINT64_MAX, this->imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        vkResetCommandBuffer(this->commandBuffers[currentFrame], 0);

        VkCommandBufferBeginInfo cmdBufferBeginInfo{};
        cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cmdBufferBeginInfo.flags = 0;
        cmdBufferBeginInfo.pInheritanceInfo = nullptr;

        if (vkBeginCommandBuffer(this->commandBuffers[currentFrame], &cmdBufferBeginInfo) != VK_SUCCESS) {
            std::cerr << "Failed to record VkBeginCommandBuffer\n";
            return;
        }

        VkImageMemoryBarrier barrier{}; // Transition of Layouts
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = this->swapChainImages[imageIndex];
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        vkCmdPipelineBarrier(
            this->commandBuffers[currentFrame],
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );


        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = this->swapChainImageViews[imageIndex];
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
        colorAttachment.clearValue = clearColor;

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea.offset = { 0, 0 };
        renderingInfo.renderArea.extent = this->extent;
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;

        vkCmdBeginRendering(this->commandBuffers[currentFrame], &renderingInfo);

        // Record draw commands here
        vkCmdBindPipeline(this->commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, this->graphicsPipeline);

        // bind vertex buffers, descriptor sets, and issue draw calls...
        VkBuffer vertexBuffers[] = { this->vertexBuffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(this->commandBuffers[currentFrame], 0, 1, vertexBuffers, offsets);
        //vkCmdDraw(this->commandBuffers[currentFrame], 6, 1, 0, 0);

        // TODO: Next uniform buffers will need to have at least as much as FRAMES_IN_FLIGHT to avoid reading/writting corruption
        const uint32_t bufferSize = 512;
        float dataBuffer[bufferSize] = { 0 };
        for (int i = 0; i < bufferSize; ++i) dataBuffer[i] = sin(i * 0.0174532925 * 10 + (rand()%10));
        memcpy(this->mappedStagedData, dataBuffer, static_cast<size_t>(bufferSize * sizeof(float)));

        transitionImageLayout(this->dataImage[this->textureIndex], VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage1D(this->stagingBuffer, this->dataImage[this->textureIndex], static_cast<uint32_t>(bufferSize));
        transitionImageLayout(this->dataImage[this->textureIndex], VK_FORMAT_R32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        VkDescriptorImageInfo descriptorImageInfo{};
        descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        descriptorImageInfo.imageView = this->dataTexture[this->textureIndex];  // Use the correct texture
        descriptorImageInfo.sampler = this->dataSampler;

        // Update descriptor set with new texture
        VkWriteDescriptorSet samplerWrite{};
        samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        samplerWrite.dstSet = this->fragDescriptorSet;
        samplerWrite.dstBinding = 1;  // Assuming texture is bound to binding 1
        samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerWrite.pImageInfo = &descriptorImageInfo;
        samplerWrite.descriptorCount = 1;
        this->textureIndex = (this->textureIndex + 1) & 1;

        vkUpdateDescriptorSets(this->device, 1, &samplerWrite, 0, nullptr);
       
        vkCmdBindDescriptorSets(
            this->commandBuffers[currentFrame],
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            this->pipelineLayout,
            0,              // First set
            1, &this->fragDescriptorSet,
            0, nullptr
        );

        vkCmdSetViewport(this->commandBuffers[currentFrame], 0, 1, &this->viewport);
        vkCmdSetScissor(this->commandBuffers[currentFrame], 0, 1, &this->scissor);

        vkCmdDraw(this->commandBuffers[currentFrame], 6, 1, 0, 0);

        vkCmdEndRendering(this->commandBuffers[currentFrame]);

        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = this->swapChainImages[imageIndex];  // <- The swapchain image used this frame
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = 0;

        vkCmdPipelineBarrier(
            this->commandBuffers[currentFrame],
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        if (vkEndCommandBuffer(this->commandBuffers[currentFrame]) != VK_SUCCESS) {
            std::cerr << "Failed to record VkEndCommandBuffer\n";
            return;
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { this->imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = this->commandBuffers + currentFrame;
        VkSemaphore signalSemaphores[] = { this->renderFinishedSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(this->graphicsQueue, 1, &submitInfo, this->inFlightFences[currentFrame]) != VK_SUCCESS) {
            std::cerr << "Failed to submit to the graphics queue on VkQueueSubmit\n";
            return;
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { this->swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        vkQueuePresentKHR(this->presentQueue, &presentInfo);

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
};