// streamsort online decoder: one Open Ephys processor for any number of data streams.
// Per stream: "enable" checkbox (stream selector) + Kilosort-folder parameter. Decoding runs on one
// thread per enabled stream (C++ core, no Python); results are published over ZMQ for the worker.
#pragma once
#include <ProcessorHeaders.h>
#include <map>
#include <memory>
#include "StreamDecoder.h"
#include "ZmqLink.h"

class StreamsortProcessor : public GenericProcessor {
public:
    StreamsortProcessor();
    ~StreamsortProcessor() override;
    AudioProcessorEditor* createEditor() override;
    void registerParameters() override;
    void updateSettings() override;
    void process(AudioBuffer<float>& buffer) override;
    void handleTTLEvent(TTLEventPtr event) override;
    bool startAcquisition() override;
    bool stopAcquisition() override;
    void parameterValueChanged(Parameter* param) override;
    String getStreamStatus(uint16 streamId);

private:
    void rebuildDecoders();
    std::map<uint16, std::unique_ptr<StreamDecoder>> decoders;
    std::map<uint16, String> status;
    std::unique_ptr<ZmqLink> zmq;
    int64 ttlCount = 0;
    std::vector<float> scratch;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(StreamsortProcessor);
};
