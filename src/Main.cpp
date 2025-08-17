#include "View.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#include "FFT.h"
#include <portaudio.h>

namespace GWaveFX {
    constexpr int SAMPLE_RATE = 44100;
    constexpr int FRAMES_PER_BUFFER = 64;

    constexpr int OUT_DEVICE = 3;
    constexpr int IN_DEVICE = 0;

    constexpr float gain = 5.0f;

    // Simple distortion effect
    float process_sample(float sample) {
        return tanhf(sample * gain);
    }

    // Callback called by PortAudio
    static int audio_callback(const void* inputBuffer, void* outputBuffer, unsigned long frameCount, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) {
        const float* in = (const float*)inputBuffer;
        float* out = (float*)outputBuffer;

        // If input is NULL (e.g., output-only stream), zero it out or skip
        if (!in) return paContinue;

        for (unsigned long i = 0; i < frameCount; i++)
            out[i] = process_sample(in[i]);

        return paContinue;
    }


    void getAudio() {
        Pa_Initialize();

        int numDevices = Pa_GetDeviceCount();
        const PaDeviceInfo* deviceInfo;

        for (int i = 0; i < numDevices; i++) {
            deviceInfo = Pa_GetDeviceInfo(i);
            printf("\n Device #%d: %s\n", i, deviceInfo->name);
            printf("   Max input channels: %d\n", deviceInfo->maxInputChannels);
            printf("   Max output channels: %d\n\n", deviceInfo->maxOutputChannels);
        }

        PaStream* stream;
        PaStreamParameters inputParams, outputParams;

        inputParams.device = Pa_GetDefaultInputDevice();
        inputParams.channelCount = 1;
        inputParams.sampleFormat = paFloat32;
        inputParams.suggestedLatency = Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
        inputParams.hostApiSpecificStreamInfo = NULL;

        outputParams.device = OUT_DEVICE;
        outputParams.channelCount = 1;
        outputParams.sampleFormat = paFloat32;
        outputParams.suggestedLatency = Pa_GetDeviceInfo(outputParams.device)->defaultLowOutputLatency;
        outputParams.hostApiSpecificStreamInfo = NULL;

        Pa_OpenStream(&stream,
            &inputParams,
            &outputParams,
            SAMPLE_RATE,
            FRAMES_PER_BUFFER,
            paClipOff,
            audio_callback,
            NULL);

        Pa_StartStream(stream);

        printf(" Running... Press Enter to stop.\n");
        getchar();

        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
    }
};

int main() {

    {
        GWaveFX::View view;
        view.run();
    }
    //GWaveFX::getAudio();

    return 0;
}