#pragma once
#include <EditorHeaders.h>

class StreamsortProcessor;

class StreamsortEditor : public GenericEditor, public Timer {
public:
    explicit StreamsortEditor(GenericProcessor* parentNode);
    ~StreamsortEditor() override {}
    void selectedStreamHasChanged() override;
    void refreshStatus();
    void timerCallback() override { refreshStatus(); }

private:
    std::unique_ptr<Label> statusLabel;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(StreamsortEditor);
};
