#include "StreamsortProcessor.h"
#include "StreamsortEditor.h"
#include <cmath>

StreamsortProcessor::StreamsortProcessor() : GenericProcessor("Streamsort") {}

StreamsortProcessor::~StreamsortProcessor() {
    for (auto& d : decoders) d.second->stop();
}

AudioProcessorEditor* StreamsortProcessor::createEditor() {
    editor = std::make_unique<StreamsortEditor>(this);
    return editor.get();
}

void StreamsortProcessor::registerParameters() {
    // per stream (shown for the stream selected in the editor); "enable_stream" is added by the GUI for filters
    addPathParameter(Parameter::STREAM_SCOPE, "ks_folder", "Kilosort folder",
                     "Kilosort output folder of this stream (decoder bank in <folder>/streamsort, made by `streamsort bank`)",
                     File(), StringArray(), true, false, true);
    // processor-wide
    addSelectedStreamParameter(Parameter::PROCESSOR_SCOPE, "stim_stream", "Stim TTL stream",
                               "Stream that carries the stimulus TTL (any stream, e.g. NI-DAQ)", {}, 0, false, true);
    addIntParameter(Parameter::PROCESSOR_SCOPE, "stim_line", "Stim TTL line", "Stim TTL line (1-based, 0 = none)", 0, 0, 256, true);
    addIntParameter(Parameter::PROCESSOR_SCOPE, "zmq_port", "ZMQ port", "Publish port (tcp://*:port)", 5557, 1024, 65535, true);
    addIntParameter(Parameter::PROCESSOR_SCOPE, "cond_port", "Condition port", "Worker condition port (tcp://127.0.0.1:port)", 5558, 1024, 65535, true);
    addIntParameter(Parameter::PROCESSOR_SCOPE, "threads", "Threads/stream", "Decoder threads per stream", 2, 1, 8, true);
}

void StreamsortProcessor::updateSettings() {
    // validate each stream's bank now so the editor can show the status before acquisition
    status.clear();
    for (auto stream : dataStreams) {
        const uint16 id = stream->getStreamId();
        if (!(bool)(*stream)["enable_stream"]) { status[id] = "disabled"; continue; }
        String folder = stream->getParameter("ks_folder")->getValueAsString();
        if (folder.isEmpty()) { status[id] = "no Kilosort folder"; continue; }
        StreamDecoder probe(stream->getName().toStdString(), id, stream->getSampleRate(), folder.toStdString(), 1, false, nullptr);
        if (probe.loadOk() && probe.channels() > stream->getChannelCount())
            status[id] = "bank needs " + String(probe.channels()) + " channels, stream has " + String(stream->getChannelCount());
        else
            status[id] = String(probe.status());
    }
}

String StreamsortProcessor::getStreamStatus(uint16 streamId) {
    auto it = status.find(streamId);
    return it == status.end() ? String("") : it->second;
}

void StreamsortProcessor::parameterValueChanged(Parameter* param) {
    if (param->getName() == "ks_folder" || param->getName() == "enable_stream") {
        updateSettings();
        if (auto* e = static_cast<StreamsortEditor*>(getEditor())) e->refreshStatus();
    }
}

void StreamsortProcessor::rebuildDecoders() {
    for (auto& d : decoders) d.second->stop();
    decoders.clear();
    const bool ttlRouted = (int)getParameter("stim_line")->getValue() > 0;
    const int threads = (int)getParameter("threads")->getValue();
    for (auto stream : dataStreams) {
        if (!(bool)(*stream)["enable_stream"]) continue;
        String folder = stream->getParameter("ks_folder")->getValueAsString();
        if (folder.isEmpty()) continue;
        auto d = std::make_unique<StreamDecoder>(stream->getName().toStdString(), stream->getStreamId(), stream->getSampleRate(),
                                                 folder.toStdString(), threads, ttlRouted, zmq.get());
        if (!d->loadOk() || d->channels() > stream->getChannelCount()) {
            LOGE("Streamsort: ", stream->getName(), ": ", String(d->status()));
            continue;
        }
        decoders[stream->getStreamId()] = std::move(d);
    }
}

bool StreamsortProcessor::startAcquisition() {
    const String pub = "tcp://*:" + String((int)getParameter("zmq_port")->getValue());
    const String cond = "tcp://127.0.0.1:" + String((int)getParameter("cond_port")->getValue());
    zmq = std::make_unique<ZmqLink>(pub.toStdString(), cond.toStdString(), [this](int c) {
        for (auto& d : decoders) d.second->setCondition(c);
    });
    if (!zmq->ok()) LOGE("Streamsort: ", String(zmq->error()));
    rebuildDecoders();
    for (auto& d : decoders) d.second->start();
    ttlCount = 0;
    LOGC("Streamsort: decoding ", (int)decoders.size(), " stream(s)");
    return true;
}

bool StreamsortProcessor::stopAcquisition() {
    for (auto& d : decoders) {
        d.second->stop();
        if (d.second->overruns()) LOGE("Streamsort: ", (int)d.second->overruns(), " blocks dropped (decoder fell > 2000 blocks behind)");
    }
    decoders.clear();
    zmq.reset();
    return true;
}

void StreamsortProcessor::handleTTLEvent(TTLEventPtr event) {
    const int line = (int)getParameter("stim_line")->getValue();
    if (line <= 0 || !event->getState() || int(event->getLine()) + 1 != line) return;
    const int sel = (int)getParameter("stim_stream")->getValue();
    if (sel < 0 || sel >= dataStreams.size() || event->getStreamId() != dataStreams[sel]->getStreamId()) return;
    const int64 k = ttlCount++;
    const double t = steadySeconds();
    for (auto& d : decoders) d.second->addTtl(k, t);
    if (zmq) {
        char h[384];
        std::snprintf(h, sizeof(h), "{\"type\":\"event\",\"stream\":\"%s\",\"line\":%d,\"state\":1,\"sample_num\":%lld,"
                      "\"timestamp_s\":%.9f,\"ttl_index\":%lld,\"t_wall_us\":%lld}",
                      jsonEscape(dataStreams[sel]->getName().toStdString()).c_str(), line, (long long)event->getSampleNumber(),
                      event->getTimestampInSeconds(), (long long)k, (long long)wallMicros());
        zmq->send("EVENT", h);
    }
}

void StreamsortProcessor::process(AudioBuffer<float>& buffer) {
    checkForEvents();                                   // stim TTLs first: they pair with this block's artifacts
    const double now = steadySeconds();
    const int64_t wall = wallMicros();
    for (auto& kv : decoders) {
        const uint16 id = kv.first;
        const int n = (int)getNumSamplesInBlock(id);
        if (n <= 0) continue;
        auto* stream = getDataStream(id);
        const int C = kv.second->channels();
        auto chans = stream->getContinuousChannels();
        Block b;
        b.s0 = getFirstSampleNumberForBlock(id); b.n = n; b.t_arrival = now; b.wall_us = wall;
        b.rows.resize(size_t(n) * C);
        for (int c = 0; c < C; ++c) {
            const float* x = buffer.getReadPointer(chans[c]->getGlobalIndex());
            const float inv = 1.0f / chans[c]->getBitVolts();
            for (int i = 0; i < n; ++i) b.rows[size_t(i) * C + c] = std::nearbyint(x[i] * inv);   // uV -> raw counts
        }
        kv.second->enqueue(std::move(b));
    }
    // data pass through unchanged
}
