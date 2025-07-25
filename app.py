from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from skimage import color, morphology
from skimage.color import deltaE_ciede2000
from sklearn.cluster import KMeans

# ───────── costanti base (invariato) ──────────
PALETTE_PATH = Path(__file__).with_name("palette_default.jpg")
VALUE_START, VALUE_STEP = 0.1, 0.2

LINES = [  # limiti A/B/C/D   y = m·x + q
    {"m": -1.4, "q": 92},
    {"m": -1.4, "q": 72},
    {"m": -1.4, "q": 52},
]

CURVE_PTS = {
    "A": [(0.2, 0.40), (0.3, 0.55), (0.5, 0.80), (0.8, 1.25),
          (1.0, 1.55), (1.2, 1.95), (1.4, 2.40), (1.5, 2.65)],
    "B": [(0.2, 0.35), (0.3, 0.50), (0.5, 0.75), (0.8, 1.20),
          (1.0, 1.45), (1.2, 1.80), (1.4, 2.20), (1.5, 2.45)],
    "C": [(0.2, 0.30), (0.3, 0.45), (0.5, 0.70), (0.8, 1.05),
          (1.0, 1.30), (1.2, 1.65), (1.4, 2.00), (1.5, 2.25)],
    "D": [(0.2, 0.25), (0.3, 0.40), (0.5, 0.60), (0.8, 0.95),
          (1.0, 1.20), (1.2, 1.50), (1.4, 1.85), (1.5, 2.10)],
}

# ───────── helper ──────────
def zone_from_TRH(T, RH):
    if RH > LINES[0]["m"] * T + LINES[0]["q"]:
        return "A"
    elif RH > LINES[1]["m"] * T + LINES[1]["q"]:
        return "B"
    elif RH > LINES[2]["m"] * T + LINES[2]["q"]:
        return "C"
    else:
        return "D"

def pressure_from_density(d, zone):
    pts = np.array(CURVE_PTS[zone])
    dens, press = pts[:, 0], pts[:, 1]
    return np.interp(d, dens, press, left=np.nan, right=np.nan)

# ───────── UI ──────────
st.set_page_config(page_title="Mappa 2-D pressione [MPa]", layout="wide")
st.title("Mappa di pressione sui pixel – visualizzazione continua")

# upload
up_img = st.file_uploader("Immagine da analizzare", ["png", "jpg", "jpeg"])
up_pal = st.file_uploader("Palette (opzionale)", ["png", "jpg", "jpeg"])

if up_img is None:
    st.stop()

# ↓ parametri
st.sidebar.header("Condizioni ambientali")
T_in = st.sidebar.number_input("Temperatura (°C)", -10.0, 60.0, 25.0, 0.5)
RH_in = st.sidebar.number_input("Umidità relativa (%)", 0.0, 100.0, 60.0, 1.0)
zone = zone_from_TRH(T_in, RH_in)
st.sidebar.write(f"→ Zona **{zone}**")

st.sidebar.header("Quantizzazione")
crop_pct  = st.sidebar.slider("Ritaglia palette (%)", 0, 40, 10)
n_col     = st.sidebar.number_input("Colori da estrarre", 1, 10, 8)
delta_thr = st.sidebar.number_input("ΔE* max", 0.0, 50.0, 18.0, 0.1)
L_white   = st.sidebar.slider("Soglia bianco L min", 80, 100, 95)
min_noise = st.sidebar.number_input("Area minima rumore", 1, 5000, 200)

st.sidebar.header("Visualizzazione")
down_perc = st.sidebar.slider("Riduzione risoluzione (%)", 10, 100, 50,
                              help="Usa 100 % per qualità massima, "
                                   "ma l'elaborazione sarà più lenta.")
colormap  = st.sidebar.selectbox("Colormap", ["Jet", "Turbo", "Viridis", "RdYlBu"],
                                 index=1)

# ───────── carica immagini ──────────
img_rgb = Image.open(up_img).convert("RGB")
arr_rgb = np.asarray(img_rgb) / 255.0
h, w, _ = arr_rgb.shape
lab_img = color.rgb2lab(arr_rgb)

pal_src = up_pal or PALETTE_PATH.open("rb")
pal_rgb = Image.open(pal_src).convert("RGB")
pal_arr = np.asarray(pal_rgb) / 255.0

# ───────── estrai palette (k-means) ──────────
ph = pal_arr.shape[0]
crop = int(ph * crop_pct / 100)
samples = color.rgb2lab(pal_arr[crop : ph - crop])
samples = samples[samples[:, :, 0] < 95].reshape(-1, 3)

cent_lab = KMeans(n_clusters=int(n_col), n_init="auto", random_state=0).fit(samples).cluster_centers_
cent_lab = cent_lab[np.argsort(cent_lab[:, 0])[::-1]]
cent_rgb = color.lab2rgb(cent_lab.reshape(1, -1, 3)).reshape(-1, 3)
values   = np.round(np.linspace(VALUE_START,
                                VALUE_START + VALUE_STEP * (len(cent_lab) - 1),
                                len(cent_lab)), 2)

# ───────── quantizzazione ΔE* ──────────
delta = np.stack(
    [deltaE_ciede2000(lab_img, c.reshape(1, 1, 3)) for c in cent_lab], axis=0
)
idx_min, delta_min = delta.argmin(axis=0), delta.min(axis=0)

white = ((lab_img[:, :, 0] >= L_white) &
         (np.abs(lab_img[:, :, 1]) + np.abs(lab_img[:, :, 2]) < 3))
valid = (delta_min <= delta_thr) & ~white
valid = morphology.remove_small_objects(valid, min_size=min_noise)
valid = morphology.binary_closing(valid, morphology.disk(1))

val_map = np.full((h, w), np.nan)
for i, v in enumerate(values):
    val_map[(idx_min == i) & valid] = v

press_map = pressure_from_density(val_map, zone)

# ───────── down-sampling per velocità ──────────
scale = down_perc / 100.0
if scale < 1.0:
    step = int(round(1 / scale))
    press_show = press_map[::step, ::step]
else:
    press_show = press_map

# ───────── mappa 2-D Plotly ──────────
st.subheader("Mappa 2-D continua della pressione [MPa]")

if np.isnan(press_show).all():
    st.info("Nessun pixel valido.")
else:
    fig = px.imshow(
        np.flipud(press_show),                # origine in basso
        origin="lower",
        color_continuous_scale=colormap,
        aspect="auto",
        labels=dict(color="Pressione [MPa]"),
        height=700, width=900
    )
    # isolinee sottili opzionali
    fig.add_contour(
        z=np.flipud(press_show),
        colorscale="Greys", showscale=False,
        line_width=0.4,
        contours=dict(start=np.nanmin(press_show),
                      end=np.nanmax(press_show), size=0.1,
                      showlabels=False)
    )
    st.plotly_chart(fig, use_container_width=True)

# ───────── download CSV opzionale ──────────
ys, xs = np.indices((h, w))
df = pd.DataFrame({
    "x_px": xs.flatten(),
    "y_px": (h - 1 - ys).flatten(),
    "Pressione": press_map.flatten()
}).dropna()

st.download_button("📥 CSV (x_px, y_px, pressione)",
                   df.to_csv(index=False).encode(),
                   "pixel_pressione.csv", "text/csv")
