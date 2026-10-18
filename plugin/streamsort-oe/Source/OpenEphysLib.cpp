#include <PluginInfo.h>
#include "StreamsortProcessor.h"
#include <string>
#ifdef WIN32
#include <Windows.h>
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

using namespace Plugin;
#define NUM_PLUGINS 1

extern "C" EXPORT void getLibInfo(Plugin::LibraryInfo* info) {
    info->apiVersion = PLUGIN_API_VER;
    info->name = "Streamsort";
    info->libVersion = "0.2.0";
    info->numPlugins = NUM_PLUGINS;
}

extern "C" EXPORT int getPluginInfo(int index, Plugin::PluginInfo* info) {
    switch (index) {
    case 0:
        info->type = Plugin::Type::PROCESSOR;
        info->processor.name = "Streamsort";
        info->processor.type = Processor::Type::FILTER;     // FILTER: per-stream enable checkbox in the stream selector
        info->processor.creator = &(Plugin::createProcessor<StreamsortProcessor>);
        break;
    default:
        return -1;
    }
    return 0;
}

#ifdef WIN32
BOOL WINAPI DllMain(IN HINSTANCE hDllHandle, IN DWORD nReason, IN LPVOID Reserved) { return TRUE; }
#endif
