#include "StreamsortEditor.h"
#include "StreamsortProcessor.h"

StreamsortEditor::StreamsortEditor(GenericProcessor* parentNode) : GenericEditor(parentNode) {
    desiredWidth = 300;
    addPathParameterEditor(Parameter::STREAM_SCOPE, "ks_folder", 10, 25);
    addSelectedStreamParameterEditor(Parameter::PROCESSOR_SCOPE, "stim_stream", 10, 50);
    addTextBoxParameterEditor(Parameter::PROCESSOR_SCOPE, "stim_line", 160, 50);
    addTextBoxParameterEditor(Parameter::PROCESSOR_SCOPE, "zmq_port", 10, 75);
    addTextBoxParameterEditor(Parameter::PROCESSOR_SCOPE, "cond_port", 110, 75);
    addTextBoxParameterEditor(Parameter::PROCESSOR_SCOPE, "threads", 210, 75);
    statusLabel = std::make_unique<Label>("status", "");
    statusLabel->setBounds(10, 103, 285, 18);
    statusLabel->setFont(FontOptions(12.0f));
    addAndMakeVisible(statusLabel.get());
    startTimer(1000);
}

void StreamsortEditor::selectedStreamHasChanged() {
    GenericEditor::selectedStreamHasChanged();
    refreshStatus();
}

void StreamsortEditor::refreshStatus() {
    auto* p = static_cast<StreamsortProcessor*>(getProcessor());
    if (statusLabel && p) statusLabel->setText(p->getStreamStatus(getCurrentStream()), dontSendNotification);
}
