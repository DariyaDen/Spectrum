import re
import numpy as np

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from PyQt5 import QtWidgets


# -----------------------------
# HPGe reference
# -----------------------------
HPGE_RESOLUTION_REF = {
    1173: 1.8,
    1332: 1.9
}


# -----------------------------
# Data container
# -----------------------------
class SpectrumData:
    def __init__(self):
        self.meta = {}
        self.channel = None
        self.energy = None
        self.counts = None
        self.rate = None
        self.calibration = None


# -----------------------------
# Parser
# -----------------------------
def parse_spectrum_file(path):
    rows = []
    meta = {}
    calib = {}

    encodings = ["utf-8", "latin-1", "cp1251"]
    text = None

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                text = f.readlines()
            break
        except:
            continue

    if text is None:
        raise ValueError("Cannot decode file")

    data_started = False

    for raw_line in text:
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            m = re.findall(r"A(\d):\s*([-0-9.eE]+)", line)
            for k, v in m:
                calib[int(k)] = float(v)

            if ":" in line:
                parts = line[1:].split(":", 1)
                if len(parts) == 2:
                    meta[parts[0].strip()] = parts[1].strip()

            if "channel data" in line.lower():
                data_started = True

            continue

        if "channel data" in line.lower():
            data_started = True
            continue

        if not data_started:
            continue

        if line.startswith("-"):
            continue

        parts = re.split(r"\s+", line)

        if len(parts) < 4:
            continue

        try:
            rows.append((float(parts[0]),
                         float(parts[1]),
                         float(parts[2]),
                         float(parts[3])))
        except:
            continue

    if len(rows) == 0:
        raise ValueError("No spectrum data parsed")

    arr = np.array(rows, dtype=float)

    spec = SpectrumData()
    spec.meta = meta
    spec.calibration = calib

    spec.channel = arr[:, 0]
    spec.energy = arr[:, 1]
    spec.counts = arr[:, 2]
    spec.rate = arr[:, 3]

    return spec


# -----------------------------
# FIT
# -----------------------------
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def fit_peak(x, y, guess_mu):
    mask = (x > guess_mu - 40) & (x < guess_mu + 40)

    x_fit = x[mask]
    y_fit = y[mask]

    if len(x_fit) < 10:
        return None

    try:
        p0 = [max(y_fit), guess_mu, 10]
        return curve_fit(gauss, x_fit, y_fit, p0=p0, maxfev=5000)[0]
    except:
        return None


def fwhm_from_sigma(sigma):
    return 2.355 * sigma


def resolution(fwhm, energy):
    return (fwhm / energy) * 100


# -----------------------------
# GUI
# -----------------------------
class SpectrumViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WinSpectrum-like Analyzer FULL")

        self.data = None

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # buttons
        self.btn_load = QtWidgets.QPushButton("Load")
        self.btn_peaks = QtWidgets.QPushButton("Peaks")
        self.btn_co60 = QtWidgets.QPushButton("Co-60")
        self.btn_zoom1332 = QtWidgets.QPushButton("Co-60 1332 Zoom")
        self.btn_save_img = QtWidgets.QPushButton("Save 1332 PNG")
        self.btn_save_csv = QtWidgets.QPushButton("Save 1332 Table")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_peaks)
        layout.addWidget(self.btn_co60)
        layout.addWidget(self.btn_zoom1332)
        layout.addWidget(self.btn_save_img)
        layout.addWidget(self.btn_save_csv)

        w = QtWidgets.QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        # connections
        self.btn_load.clicked.connect(self.load_file)
        self.btn_peaks.clicked.connect(self.peaks)
        self.btn_co60.clicked.connect(self.analyze_co60)
        self.btn_zoom1332.clicked.connect(self.show_peak2)
        self.btn_save_img.clicked.connect(self.save_image)
        self.btn_save_csv.clicked.connect(self.save_table)

        self.roi_selector = SpanSelector(
            self.ax,
            self.on_roi,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.3, facecolor="red")
        )

    # -----------------------------
    def load_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open")
        if path:
            self.data = parse_spectrum_file(path)
            self.plot()

    def plot(self):
        self.ax.clear()
        self.ax.plot(self.data.energy, self.data.counts, lw=1)
        self.canvas.draw()

    # -----------------------------
    def peaks(self):
        peaks, _ = find_peaks(self.data.counts, height=max(self.data.counts)*0.05)

        self.ax.clear()
        self.ax.plot(self.data.energy, self.data.counts)
        self.ax.plot(self.data.energy[peaks], self.data.counts[peaks], "ro")
        self.canvas.draw()

    # -----------------------------
    def analyze_co60(self):
        self.ax.clear()
        self.ax.plot(self.data.energy, self.data.counts)

        for E in [1173, 1332]:
            fit = fit_peak(self.data.energy, self.data.counts, E)
            if fit is None:
                continue
            A, mu, sigma = fit
            self.ax.axvline(mu, color="red")

        self.canvas.draw()

    # -----------------------------
    # 1332 zoom
    # -----------------------------
    def show_peak2(self):
        x = self.data.energy
        y = self.data.counts

        target = 1332
        mask = (x > target - 80) & (x < target + 80)

        xz = x[mask]
        yz = y[mask]

        fit = fit_peak(x, y, target)

        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        ax.plot(xz, yz)

        if fit is not None:
            A, mu, sigma = fit

            self.last_peak = {
                "mu": mu,
                "sigma": sigma,
                "fwhm": fwhm_from_sigma(sigma),
                "res": resolution(fwhm_from_sigma(sigma), mu),
                "area": np.trapz(yz, xz),
                "height": A,
                "xmin": xz.min(),
                "xmax": xz.max()
            }

            xf = np.linspace(xz.min(), xz.max(), 300)
            ax.plot(xf, gauss(xf, A, mu, sigma), "r--")

        win = QtWidgets.QMainWindow()
        win.setCentralWidget(canvas)
        win.resize(700, 450)
        win.show()

        self.zoom_win = win

    # -----------------------------
    def save_image(self):
        if not hasattr(self, "zoom_win"):
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PNG", "peak.png")
        if path:
            self.zoom_win.grab().save(path)

    # -----------------------------
    def save_table(self):
        if not hasattr(self, "last_peak"):
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "peak.csv")
        if not path:
            return

        d = self.last_peak

        with open(path, "w") as f:
            f.write("parameter,value\n")
            for k, v in d.items():
                f.write(f"{k},{v}\n")

    # -----------------------------
    def on_roi(self, xmin, xmax):
        pass


# -----------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    w = SpectrumViewer()
    w.resize(1000, 700)
    w.show()
    app.exec_()
