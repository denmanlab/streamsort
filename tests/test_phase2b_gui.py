"""Phase 2b: worker GUI state machine and displays (Qt offscreen)."""
import os
import numpy as np, pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
zmq = pytest.importorskip("zmq")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_gui_configure_arm_run_stop(app, tmp_path):
    from streamsort.worker2.gui import WorkerWindow
    from streamsort.worker2.analyses import make_placeholder_images
    make_placeholder_images(2, tmp_path / "img")
    ctx = zmq.Context()
    w = WorkerWindow(dict(pub="inproc://g-pub", cond="inproc://g-cond", conditions=[dict(a=0), dict(a=1)], schedule=[0, 1],
                          images_dir=str(tmp_path / "img")), context=ctx)
    assert w.state == "CONFIGURE" and w.form_box.isEnabled()
    ec = w.engine_config(w.form_config()); assert ec["analysis"] == "procrustes" and ec["schedule"] == [0, 1]
    w.arm(); assert w.state == "ARMED" and not w.form_box.isEnabled()
    tgt = np.random.default_rng(0).standard_normal((50, 3))
    w._on_trial(dict(k=0, ttl_to_loss_ms=530.0, loss=None), dict(state="armed", target=tgt))
    w._on_trial(dict(k=1, ttl_to_loss_ms=531.0, loss=0.3), dict(state="running", target=tgt, trial=tgt + 0.1))
    w._on_status(dict(kind="stim", stim="programmed", ttl_to_stim_ms=150.0))
    assert w.state == "RUNNING" and len(w.loss) == 1 and len(w.lat_stim) == 1
    w.stop(); assert w.state == "CONFIGURE"
    w.c_analysis.setCurrentText("image_decoder"); w.arm(); assert w.stack.currentIndex() == 1
    w._on_trial(dict(k=0, ttl_to_loss_ms=530.0, loss=0.2), dict(state="running", probs=np.array([.2, .8]), classes=[0, 1], decoded=1,
                                                                 decoded_image=str(tmp_path / "img" / "image_01.png"),
                                                                 target_image=str(tmp_path / "img" / "image_01.png")))
    assert w.img_dec.pixmap() is not None and not w.img_dec.pixmap().isNull()
    w.stop(); w.close(); ctx.term()
