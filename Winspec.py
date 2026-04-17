import re
import numpy as np

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5 import QtWidgets


# -----------------------------
# DATA
# -----------------------------
class SpectrumData:
    def __init__(self):
        self.energy = None
        self.counts = None


# -----------------------------
# PARSER
# -----------------------------
def parse_spectrum_file(path):

    rows = []
    text = None

    for enc in ["utf-8", "latin-1", "cp1251"]:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                text = f.readlines()
            break
        except:
            continue

    if text is None:
        raise ValueError("Cannot read file")

    data_started = False

    for line in text:
        s = line.strip()
        if not s:
            continue

        if re.match(r"^\s*\d+", s):
            data_started = True

        if not data_started:
            continue

        parts = re.split(r"\s+", s)

        try:
            nums = list(map(float, parts))
            if len(nums) >= 3:
                rows.append(nums[:3])
        except:
            continue

    if len(rows) == 0:
        raise ValueError("No spectrum data parsed")

    arr = np.array(rows)

    spec = SpectrumData()
    spec.energy = arr[:, 1]
    spec.counts = arr[:, 2]

    return spec


# -----------------------------
# MODEL
# -----------------------------
def model(x, A, mu, sigma, c0, c1):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c0 + c1 * x


# -----------------------------
# FIT (FIXED: adaptive ROI)
# -----------------------------
def fit_peak_adaptive(x, y, mu0):

    # initial fit window
    mask = (x > mu0 - 20) & (x < mu0 + 20)
    x0 = x[mask]
    y0 = y[mask]

    if len(x0) < 10:
        return None

    p0 = [max(y0), mu0, 1.2, np.median(y0), 0.0]

    try:
        popt, _ = curve_fit(model, x0, y0, p0=p0, maxfev=10000)
    except:
        return None

    A, mu, sigma, c0, c1 = popt

    # adaptive ROI (CRITICAL FIX)
    roi = max(8.0, 3.0 * sigma)
    mask = (x > mu - roi) & (x < mu + roi)

    x1 = x[mask]
    y1 = y[mask]

    if len(x1) < 10:
        return popt

    try:
        popt2, _ = curve_fit(model, x1, y1, p0=popt, maxfev=10000)
        return popt2
    except:
        return popt


# -----------------------------
# METRICS
# -----------------------------
def fwhm(sigma):
    return 2.35482 * sigma


def resolution(fwhm_val, energy):
    return (fwhm_val / energy) * 100


# -----------------------------
# GUI
# -----------------------------
class SpectrumViewer(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HPGe Analyzer (Full Feature Set + Stable Fit)")

        self.data = None
        self.last_peak = None
        self.zoom_win = None

        # canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # ---------------- BUTTONS ----------------
        self.btn_load = QtWidgets.QPushButton("Load spectrum")
        self.btn_full = QtWidgets.QPushButton("Show full spectrum")
        self.btn_peaks = QtWidgets.QPushButton("Find peaks")

        self.btn_1332 = QtWidgets.QPushButton("Co-60 1332 keV peak")

        self.btn_save_full = QtWidgets.QPushButton("Save full spectrum image")
        self.btn_save_peak = QtWidgets.QPushButton("Save 1332 peak image")
        self.btn_save_csv = QtWidgets.QPushButton("Save peak CSV")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)

        # ALL BUTTONS RESTORED
        for b in [
            self.btn_load,
            self.btn_full,
            self.btn_peaks,
            self.btn_1332,
            self.btn_save_full,
            self.btn_save_peak,
            self.btn_save_csv
        ]:
            layout.addWidget(b)

        w = QtWidgets.QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        # ---------------- CONNECTIONS ----------------
        self.btn_load.clicked.connect(self.load)
        self.btn_full.clicked.connect(self.plot_full)
        self.btn_peaks.clicked.connect(self.find_peaks)
        self.btn_1332.clicked.connect(self.show_1332)

        self.btn_save_full.clicked.connect(self.save_full)
        self.btn_save_peak.clicked.connect(self.save_peak)
        self.btn_save_csv.clicked.connect(self.save_csv)

    # -----------------------------
    def load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open spectrum")
        if path:
            self.data = parse_spectrum_file(path)
            self.plot_full()

    def plot_full(self):
        self.ax.clear()
        self.ax.plot(self.data.energy, self.data.counts, lw=1)
        self.ax.set_xlabel("Energy (keV)")
        self.ax.set_ylabel("Counts")
        self.canvas.draw()

    # -----------------------------
    def find_peaks(self):
        peaks, _ = find_peaks(self.data.counts, height=max(self.data.counts)*0.05)

        self.ax.clear()
        self.ax.plot(self.data.energy, self.data.counts)
        self.ax.plot(self.data.energy[peaks], self.data.counts[peaks], "ro")
        self.canvas.draw()

    # -----------------------------
    def show_1332(self):

        if self.data is None:
            return

        popt = fit_peak_adaptive(self.data.energy, self.data.counts, 1332)

        if popt is None:
            return

        A, mu, sigma, c0, c1 = popt

        roi = max(8.0, 3.0 * sigma)
        mask = (self.data.energy > mu - roi) & (self.data.energy < mu + roi)

        x = self.data.energy[mask]
        y = self.data.counts[mask]

        fig = Figure()
        ax = fig.add_subplot(111)

        ax.plot(x, y, lw=1)

        xf = np.linspace(x.min(), x.max(), 400)
        ax.plot(xf, model(xf, *popt), "r--")

        fwhm_val = fwhm(sigma)
        res = resolution(fwhm_val, mu)

        self.last_peak = {
            "mu": mu,
            "sigma": sigma,
            "fwhm": fwhm_val,
            "res_%": res,
            "height": A,
            "xmin": x.min(),
            "xmax": x.max()
        }

        print("\n1332 keV peak:")
        print(f"mu = {mu:.3f}")
        print(f"sigma = {sigma:.3f}")
        print(f"fwhm = {fwhm_val:.3f}")
        print(f"res = {res:.3f}%")

        win = QtWidgets.QMainWindow()
        canvas = FigureCanvas(fig)
        win.setCentralWidget(canvas)
        win.resize(800, 500)
        win.show()

        self.zoom_win = win

    # -----------------------------
    def save_full(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save full spectrum", "spectrum.png")
        if path:
            self.figure.savefig(path, dpi=300, bbox_inches="tight")

    # -----------------------------
    def save_peak(self):
        if self.zoom_win is None:
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save peak image", "peak.png")
        if path:
            self.zoom_win.grab().save(path)

    # -----------------------------
    def save_csv(self):

        if self.last_peak is None:
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "peak.csv")
        if not path:
            return

        with open(path, "w") as f:
            f.write("parameter,value\n")
            for k, v in self.last_peak.items():
                f.write(f"{k},{v}\n")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    w = SpectrumViewer()
    w.resize(1000, 700)
    w.show()
    app.exec_()
