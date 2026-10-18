import os, sys, dataclasses
import numpy as np, pytest
ROOT_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_REPO)
DATA = "/Users/danieldenman/ephys_data_stim/jlh32_2023-02-18_16-07-58"
RESULTS = os.path.join(ROOT_REPO, "results")
needs_data = pytest.mark.skipif(not os.path.isdir(DATA), reason="jlh32 recording not mounted")
needs_results = pytest.mark.skipif(not os.path.isfile(os.path.join(RESULTS, "templates_jlh32_probeA.npz")),
                                   reason="template bank / results not present")

@pytest.fixture(scope="session")
def stream():
    from streamsort.io import OpenEphysSession
    sess = OpenEphysSession(DATA + "/Record Node 105/experiment1/recording1")
    return sess.stream("Neuropix-PXI-104.ProbeA-AP")

@pytest.fixture(scope="session")
def cfg(stream):
    from streamsort.config import StreamConfig
    return dataclasses.replace(StreamConfig(sample_rate=stream.effective_rate, bit_volts=stream.bit_volts), detect_threshold=4.0)

@pytest.fixture(scope="session")
def src(stream):
    from streamsort.io import FileStreamSource
    return FileStreamSource(stream, 200)

@pytest.fixture(scope="session")
def trials():
    import pandas as pd
    return pd.read_csv(os.path.join(RESULTS, "jlh32_probeA_trials.csv"))

@pytest.fixture(scope="session")
def decoder_parts():
    import pandas as pd
    from streamsort.sorting.templates import TemplateBank
    from streamsort.online import Whitening
    bank = TemplateBank.load(os.path.join(RESULTS, "templates_jlh32_probeA.npz"))
    usable = pd.read_csv(os.path.join(RESULTS, "sorting_report.csv")).usable.to_numpy(bool)
    wh = Whitening.load(os.path.join(RESULTS, "whitening_probeA.npz"))
    return bank, usable, wh
