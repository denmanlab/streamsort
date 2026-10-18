"""streamsort worker GUI (PySide6 + pyqtgraph).

States: CONFIGURE (edit reducer / loss / target / stimulator / conditions) -> ARMED (connected to the
plugin; shows the target once it exists) -> RUNNING (per trial: the trial's latents next to the target,
or the decoded image next to the target image; loss history; TTL -> loss and TTL -> stimulator latency).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .engine import Engine, ZmqWorker
from . import protocol

pg.setConfigOptions(antialias=True, background="w", foreground="k")
PC_COLORS = [(44, 79, 143), (181, 101, 29), (58, 122, 58)]


class Bridge(QtCore.QObject):
    trial = QtCore.Signal(object, object)
    status = QtCore.Signal(object)


DEFAULTS = dict(pub=protocol.DEFAULT_PUB, cond=protocol.DEFAULT_COND, iti_s=2.0, win_ms=500.0, bin_ms=10.0, streams=[],
                analysis="procrustes", reducer="pca", k=3, n_fit=20, target_condition=None, target_file="",
                images_dir="", target_class=0, readout="", stimulator="dummy", port="", baud=115200,
                command="SET AMP {amplitude} POL {polarity} DUR {pulse_duration} NEG {contact_negative} POS {contact_positive}\\n",
                reprog_ms=100.0, conditions=[], schedule=[], log="")


class WorkerWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: dict | None = None, context=None):
        super().__init__()
        self.setWindowTitle("streamsort worker")
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        self.ctx = context; self.engine = None; self.zw = None; self.state = "CONFIGURE"
        self.bridge = Bridge(); self.bridge.trial.connect(self._on_trial); self.bridge.status.connect(self._on_status)
        self.loss, self.lat_loss, self.lat_stim, self.holds = [], [], [], 0
        self._build()
        self._load_into_form(self.cfg)
        self._set_state("CONFIGURE")

    # -- layout ------------------------------------------------------------------------------
    def _build(self):
        w = QtWidgets.QWidget(); self.setCentralWidget(w); h = QtWidgets.QHBoxLayout(w)
        left = QtWidgets.QVBoxLayout(); h.addLayout(left, 0)
        self.form_box = QtWidgets.QGroupBox("Configure"); f = QtWidgets.QFormLayout(self.form_box); left.addWidget(self.form_box)
        self.e_pub = QtWidgets.QLineEdit(); f.addRow("Plugin publisher", self.e_pub)
        self.e_cond = QtWidgets.QLineEdit(); f.addRow("Condition socket", self.e_cond)
        self.e_streams = QtWidgets.QLineEdit(); self.e_streams.setPlaceholderText("all streams that say hello"); f.addRow("Streams", self.e_streams)
        self.s_iti = QtWidgets.QDoubleSpinBox(); self.s_iti.setRange(0.1, 60); self.s_iti.setSuffix(" s"); f.addRow("ITI", self.s_iti)
        self.s_win = QtWidgets.QDoubleSpinBox(); self.s_win.setRange(10, 5000); self.s_win.setSuffix(" ms"); f.addRow("Window", self.s_win)
        self.s_bin = QtWidgets.QDoubleSpinBox(); self.s_bin.setRange(1, 100); self.s_bin.setSuffix(" ms"); f.addRow("Bin", self.s_bin)
        self.c_analysis = QtWidgets.QComboBox(); self.c_analysis.addItems(["procrustes", "image_decoder"]); f.addRow("Loss", self.c_analysis)
        self.c_reducer = QtWidgets.QComboBox(); self.c_reducer.addItems(["pca", "fa", "isomap"]); f.addRow("Reducer", self.c_reducer)
        self.s_k = QtWidgets.QSpinBox(); self.s_k.setRange(2, 20); f.addRow("Latent dims", self.s_k)
        self.s_nfit = QtWidgets.QSpinBox(); self.s_nfit.setRange(2, 1000); f.addRow("Fit trials", self.s_nfit)
        self.c_target = QtWidgets.QComboBox(); f.addRow("Target condition", self.c_target)
        self.e_target_file = QtWidgets.QLineEdit(); self.e_target_file.setPlaceholderText("optional .npy (n_bins x k)"); f.addRow("Target file", self.e_target_file)
        self.e_images = QtWidgets.QLineEdit(); self.e_images.setPlaceholderText("folder of image_XX.png (class = condition id)"); f.addRow("Image set", self.e_images)
        self.e_readout = QtWidgets.QLineEdit(); self.e_readout.setPlaceholderText("optional readout .npz (else nearest centroid)"); f.addRow("Readout", self.e_readout)
        self.c_stim = QtWidgets.QComboBox(); self.c_stim.addItems(["dummy", "serial"]); f.addRow("Stimulator", self.c_stim)
        self.e_port = QtWidgets.QLineEdit(); f.addRow("Serial port", self.e_port)
        self.e_cmd = QtWidgets.QLineEdit(); f.addRow("Command", self.e_cmd)
        self.s_reprog = QtWidgets.QDoubleSpinBox(); self.s_reprog.setRange(0, 5000); self.s_reprog.setSuffix(" ms"); f.addRow("Reprogramming", self.s_reprog)
        self.e_conditions = QtWidgets.QLineEdit(); self.e_conditions.setPlaceholderText("conditions JSON file"); f.addRow("Conditions", self.e_conditions)
        row = QtWidgets.QHBoxLayout()
        self.b_load = QtWidgets.QPushButton("Load…"); self.b_save = QtWidgets.QPushButton("Save…")
        self.b_arm = QtWidgets.QPushButton("Arm"); self.b_stop = QtWidgets.QPushButton("Stop")
        for b in (self.b_load, self.b_save, self.b_arm, self.b_stop):
            row.addWidget(b)
        left.addLayout(row)
        self.b_load.clicked.connect(self._load_dialog); self.b_save.clicked.connect(self._save_dialog)
        self.b_arm.clicked.connect(self.arm); self.b_stop.clicked.connect(self.stop)
        self.c_analysis.currentTextChanged.connect(lambda t: self.stack.setCurrentIndex(0 if t == "procrustes" else 1))
        self.lbl_state = QtWidgets.QLabel(); self.lbl_state.setStyleSheet("font-weight: bold; font-size: 14px"); left.addWidget(self.lbl_state)
        self.lbl_info = QtWidgets.QLabel(); self.lbl_info.setWordWrap(True); left.addWidget(self.lbl_info); left.addStretch(1)

        right = QtWidgets.QVBoxLayout(); h.addLayout(right, 1)
        self.stack = QtWidgets.QStackedWidget(); right.addWidget(self.stack, 3)
        # procrustes view
        self.glw = pg.GraphicsLayoutWidget()
        self.p_time = self.glw.addPlot(title="latents vs time: target (dashed) and trial (solid)")
        self.p_time.setLabel("bottom", "time after pulse", units="ms"); self.p_time.addLegend(offset=(5, 5))
        self.p_plane = self.glw.addPlot(title="latent plane (PC1 × PC2)"); self.p_plane.setAspectLocked(False)
        self.stack.addWidget(self.glw)
        # decoder view
        dv = QtWidgets.QWidget(); dl = QtWidgets.QHBoxLayout(dv)
        self.img_target = QtWidgets.QLabel("target"); self.img_dec = QtWidgets.QLabel("decoded")
        for lbl, title in ((self.img_target, "target image"), (self.img_dec, "decoded image")):
            box = QtWidgets.QGroupBox(title); bl = QtWidgets.QVBoxLayout(box); lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setMinimumSize(180, 180); bl.addWidget(lbl); dl.addWidget(box, 1)
        self.p_prob = pg.PlotWidget(title="class probabilities"); self.p_prob.setMinimumWidth(300); dl.addWidget(self.p_prob, 2)
        self.stack.addWidget(dv)
        # history
        hist = pg.GraphicsLayoutWidget(); right.addWidget(hist, 2)
        self.p_loss = hist.addPlot(title="loss per trial"); self.p_loss.setLabel("bottom", "trial")
        self.c_loss = self.p_loss.plot(pen=None, symbol="o", symbolSize=5, symbolBrush=(44, 79, 143))
        self.p_lat = hist.addPlot(title="latency after stim TTL"); self.p_lat.setLabel("bottom", "trial"); self.p_lat.setLabel("left", "ms")
        self.p_lat.addLegend(offset=(5, 5))
        self.c_lat_loss = self.p_lat.plot(pen=None, symbol="o", symbolSize=5, symbolBrush=(44, 79, 143), name="loss ready")
        self.c_lat_stim = self.p_lat.plot(pen=None, symbol="t", symbolSize=6, symbolBrush=(181, 101, 29), name="stimulator programmed")
        self.iti_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("k", style=QtCore.Qt.DashLine)); self.p_lat.addItem(self.iti_line)
        self.resize(1400, 820)

    # -- config <-> form ------------------------------------------------------------------------
    def _load_into_form(self, c):
        self.e_pub.setText(c["pub"]); self.e_cond.setText(c["cond"]); self.e_streams.setText(",".join(c["streams"]))
        self.s_iti.setValue(c["iti_s"]); self.s_win.setValue(c["win_ms"]); self.s_bin.setValue(c["bin_ms"])
        self.c_analysis.setCurrentText(c["analysis"]); self.stack.setCurrentIndex(0 if c["analysis"] == "procrustes" else 1); self.c_reducer.setCurrentText(c["reducer"]); self.s_k.setValue(int(c["k"]))
        self.s_nfit.setValue(int(c["n_fit"])); self.e_target_file.setText(c["target_file"]); self.e_images.setText(c["images_dir"])
        self.e_readout.setText(c["readout"]); self.c_stim.setCurrentText(c["stimulator"]); self.e_port.setText(c["port"])
        self.e_cmd.setText(c["command"]); self.s_reprog.setValue(c["reprog_ms"])
        self.e_conditions.setText(c.get("conditions_file", ""))
        self.c_target.clear(); self.c_target.addItem("(all trials)", None)
        for i, cd in enumerate(c["conditions"]):
            self.c_target.addItem(f"{i}: " + ", ".join(f"{k}={v}" for k, v in cd.items()), i)
        tc = c.get("target_condition")
        if tc is not None:
            self.c_target.setCurrentIndex(int(tc) + 1)

    def form_config(self) -> dict:
        c = dict(self.cfg)
        c.update(pub=self.e_pub.text(), cond=self.e_cond.text(), streams=[s for s in self.e_streams.text().split(",") if s.strip()],
                 iti_s=self.s_iti.value(), win_ms=self.s_win.value(), bin_ms=self.s_bin.value(), analysis=self.c_analysis.currentText(),
                 reducer=self.c_reducer.currentText(), k=self.s_k.value(), n_fit=self.s_nfit.value(), target_condition=self.c_target.currentData(),
                 target_file=self.e_target_file.text(), images_dir=self.e_images.text(), readout=self.e_readout.text(),
                 stimulator=self.c_stim.currentText(), port=self.e_port.text(), command=self.e_cmd.text(), reprog_ms=self.s_reprog.value())
        cf = self.e_conditions.text().strip()
        if cf and Path(cf).exists():
            d = json.loads(Path(cf).read_text()); c["conditions"] = d.get("conditions", d) if isinstance(d, dict) else d
            if isinstance(d, dict) and "schedule" in d:
                c["schedule"] = d["schedule"]
            c["conditions_file"] = cf
        return c

    def engine_config(self, c) -> dict:
        images = {}
        if c["images_dir"]:
            for p in sorted(Path(c["images_dir"]).glob("image_*.png")):
                images[int(p.stem.split("_")[-1])] = str(p)
        akw = (dict(reducer=c["reducer"], k=c["k"], n_fit=c["n_fit"], target_condition=c["target_condition"],
                    target=c["target_file"] or None) if c["analysis"] == "procrustes" else
               dict(images=images, target_class=c["target_condition"] if c["target_condition"] is not None else c["target_class"],
                    n_fit=c["n_fit"], model=c["readout"] or None))
        skw = dict(reprog_ms=c["reprog_ms"]) if c["stimulator"] == "dummy" else dict(port=c["port"], baud=c["baud"], command=c["command"].encode().decode("unicode_escape"), reprog_ms=c["reprog_ms"])
        return dict(streams=c["streams"], iti_s=c["iti_s"], win_ms=c["win_ms"], bin_ms=c["bin_ms"], analysis=c["analysis"], analysis_kw=akw,
                    stimulator=c["stimulator"], stimulator_kw=skw, conditions=c["conditions"],
                    schedule=c["schedule"] or list(range(len(c["conditions"]))), log=c["log"])

    def _load_dialog(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load worker config", "", "JSON (*.json)")
        if fn:
            self.cfg = dict(DEFAULTS, **json.loads(Path(fn).read_text())); self._load_into_form(self.cfg)

    def _save_dialog(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save worker config", "worker_gui.json", "JSON (*.json)")
        if fn:
            Path(fn).write_text(json.dumps(self.form_config(), indent=1))

    # -- state machine ---------------------------------------------------------------------------
    def _set_state(self, s):
        self.state = s
        self.lbl_state.setText({"CONFIGURE": "● Configure", "ARMED": "● Armed — waiting for target",
                                "RUNNING": "● Running"}[s])
        self.form_box.setEnabled(s == "CONFIGURE"); self.b_arm.setEnabled(s == "CONFIGURE"); self.b_stop.setEnabled(s != "CONFIGURE")

    def arm(self):
        c = self.form_config(); self.cfg = c
        self.loss, self.lat_loss, self.lat_stim, self.holds = [], [], [], 0
        self.stack.setCurrentIndex(0 if c["analysis"] == "procrustes" else 1)
        self.iti_line.setValue(c["iti_s"] * 1e3)
        self.engine = Engine(self.engine_config(c), on_trial=lambda r, d: self.bridge.trial.emit(r, d),
                             on_status=lambda s: self.bridge.status.emit(s))
        self.zw = ZmqWorker(self.engine, pub=c["pub"], cond=c["cond"], context=self.ctx).start()
        self.engine.arm()
        self._set_state("ARMED"); self._info()

    def stop(self):
        if self.zw:
            self.zw.stop()
        if self.engine:
            self.engine.close()
        self.zw = self.engine = None
        self._set_state("CONFIGURE")

    def closeEvent(self, e):
        self.stop(); super().closeEvent(e)

    # -- updates ---------------------------------------------------------------------------------
    def _info(self):
        e = self.engine
        n = len(e.rows) if e else 0
        streams = ", ".join(f"{k} ({len(v.units)} units)" for k, v in (e.S.items() if e else []))
        self.lbl_info.setText(f"streams: {streams or '—'}\ntrials: {n}   holds: {self.holds}\n"
                              f"messages: {self.zw.n_msgs if self.zw else 0}")

    def _on_status(self, s):
        if s.get("kind") == "stim":
            self.lat_stim.append(s["ttl_to_stim_ms"]); self.holds += s["stim"] == "hold"
            self.c_lat_stim.setData(np.arange(len(self.lat_stim)), np.asarray(self.lat_stim))
        self._info()

    def _on_trial(self, row, disp):
        st = disp.get("state")
        if st in ("armed", "running") and self.state == "ARMED":
            self._set_state("RUNNING" if st == "running" else "ARMED")
            self.lbl_state.setText("● Armed — target ready" if st == "armed" else self.lbl_state.text())
        if st == "running":
            self._set_state("RUNNING")
        self.lat_loss.append(row["ttl_to_loss_ms"]); self.c_lat_loss.setData(np.arange(len(self.lat_loss)), np.asarray(self.lat_loss))
        if row["loss"] is not None:
            self.loss.append((len(self.lat_loss) - 1, row["loss"]))
            x, y = zip(*self.loss); self.c_loss.setData(np.asarray(x), np.asarray(y))
        if self.cfg["analysis"] == "procrustes" and "target" in disp:
            self._draw_latents(disp["target"], disp.get("trial"))
        elif self.cfg["analysis"] == "image_decoder":
            self._draw_images(disp)
        self._info()

    def _draw_latents(self, tgt, trial):
        bins = (np.arange(len(tgt)) + 0.5) * self.cfg["bin_ms"] + 2.0
        self.p_time.clear(); self.p_plane.clear()
        for i in range(min(3, tgt.shape[1])):
            c = PC_COLORS[i]
            self.p_time.plot(bins, tgt[:, i], pen=pg.mkPen(c, width=2, style=QtCore.Qt.DashLine), name=f"PC{i + 1}")
            if trial is not None:
                self.p_time.plot(bins, trial[:, i], pen=pg.mkPen(c, width=2))
        self.p_plane.plot(tgt[:, 0], tgt[:, 1], pen=pg.mkPen((90, 90, 90), width=2, style=QtCore.Qt.DashLine))
        if trial is not None:
            self.p_plane.plot(trial[:, 0], trial[:, 1], pen=pg.mkPen((44, 79, 143), width=2))

    def _set_img(self, lbl, path):
        if path and Path(path).exists():
            lbl.setPixmap(QtGui.QPixmap(path).scaled(220, 220, QtCore.Qt.KeepAspectRatio))
        else:
            lbl.setText("—")

    def _draw_images(self, d):
        self._set_img(self.img_target, d.get("target_image"))
        if d.get("state") == "running":
            self._set_img(self.img_dec, d.get("decoded_image"))
            self.p_prob.clear()
            p = np.asarray(d["probs"]); x = np.arange(len(p))
            brushes = [(181, 101, 29) if c == d["decoded"] else (150, 150, 150) for c in d["classes"]]
            self.p_prob.addItem(pg.BarGraphItem(x=x, height=p, width=0.8, brushes=brushes))
            step = max(1, len(d["classes"]) // 6)
            self.p_prob.getAxis("bottom").setTicks([[(i, str(c)) for i, c in enumerate(d["classes"]) if i % step == 0]])
            self.p_prob.setLabel("bottom", "class (condition id); orange = decoded")


def main(cfg_path: str | None = None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cfg = json.loads(Path(cfg_path).read_text()) if cfg_path else None
    w = WorkerWindow(cfg); w.show()
    return app.exec()
