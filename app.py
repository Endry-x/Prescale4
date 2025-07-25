from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
from skimage import color, morphology
from skimage.color import deltaE_ciede2000
from sklearn.cluster import KMeans
import importlib

# ───────── costanti
PALETTE_PATH = Path(__file__).with_name("palette_default.jpg")
VALUE_START, VALUE_STEP = 0.1, 0.2

LINES = [
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

# ───────── helper
def zone_from_TRH(T, RH):
    if RH > LINES[0]["m"]*T + LINES[0]["q"]:
        return "A"
    elif RH > LINES[1]["m"]*T + LINES[1]["q"]:
        return "B"
    elif RH > LINES[2]["m"]*T + LINES[2]["q"]:
        return "C"
    else:
        return "D"

def pressure_from_density(d, zone):
    pts = np.array(CURVE_PTS[zone])
    return np.interp(d, pts[:, 0], pts[:, 1], left=np.nan, right=np.nan)

# ───────── UI
st.set_page_config(page_title="Mappa pressione 2-D", layout="wide")
st.title("Campo di pressione [MPa] sui pixel")

up_img = st.file_uploader("Immagine da analizzare", ["png", "jpg", "jpeg"])
up_pal = st.file_uploader("Palette (opz.)", ["png", "jpg", "jpeg"])
if up_img is None:
    st.stop()

# condizioni
st.sidebar.header("Condizioni")
T = st.sidebar.number_input("Temperatura (°C)", -10.0, 60.0, 25.0)
RH = st.sidebar.number_input("Umidità relativa (%)", 0.0, 100.0, 60.0)
zone = zone_from_TRH(T, RH)
st.sidebar.write(f"→ Zona **{zone}**")

# quantizzazione
st.sidebar.header("Quantizzazione")
crop_pct  = st.sidebar.slider("Ritaglio palette (%)", 0, 40, 10)
k_clust   = st.sidebar.number_input("Cluster K-means", 1, 10, 8)
delta_thr = st.sidebar.slider("ΔE* max", 0.0, 50.0, 25.0)
L_white   = st.sidebar.slider("Soglia bianco L", 80, 100, 95)
min_noise = st.sidebar.number_input("Area min rumore", 1, 5000, 200)

# display
st.sidebar.header("Display")
down_perc = st.sidebar.slider("Riduzione risoluzione (%)", 10, 100, 60)
cmap      = st.sidebar.selectbox("Colormap", ["Turbo", "Jet", "Viridis", "RdYlBu"], 0)
sigma     = st.sidebar.slider("Blur (σ px)", 0, 5, 1)

# ───────── carica immagini
img_rgb = Image.open(up_img).convert("RGB")
arr_rgb = np.asarray(img_rgb)/255.0
h, w, _ = arr_rgb.shape
lab_img = color.rgb2lab(arr_rgb)

pal_rgb = Image.open(up_pal or PALETTE_PATH).convert("RGB")
pal_arr = np.asarray(pal_rgb)/255.0
crop = int(pal_arr.shape[0]*crop_pct/100)
samples = color.rgb2lab(pal_arr[crop:-crop])
samples = samples[samples[:, :, 0] < 95].reshape(-1, 3)

# ───────── K-means palette
cent_lab = KMeans(k_clust, n_init="auto", random_state=0).fit(samples).cluster_centers_
cent_lab = cent_lab[np.argsort(cent_lab[:,0])[::-1]]
cent_rgb = color.lab2rgb(cent_lab.reshape(1,-1,3)).reshape(-1,3)
values   = np.round(np.linspace(VALUE_START, VALUE_START+VALUE_STEP*(k_clust-1), k_clust), 2)

# ΔE*
delta = np.stack([deltaE_ciede2000(lab_img, c.reshape(1,1,3)) for c in cent_lab], 0)
idx, dmin = delta.argmin(0), delta.min(0)

white = ((lab_img[:,:,0] >= L_white) &
         (np.abs(lab_img[:,:,1]) + np.abs(lab_img[:,:,2]) < 3))
mask = (dmin <= delta_thr) & ~white
mask = morphology.remove_small_objects(mask, min_noise)
mask = morphology.binary_closing(mask, morphology.disk(1))

dens = np.full((h, w), np.nan)
for i, v in enumerate(values):
    dens[(idx == i) & mask] = v
press = pressure_from_density(dens, zone)

# immagine quantizzata
quant_rgb = arr_rgb.copy()
for i, col in enumerate(cent_rgb):
    quant_rgb[(idx == i) & mask] = col

# blur opzionale
if sigma>0 and importlib.util.find_spec("scipy"):
    from scipy.ndimage import gaussian_filter
    press = gaussian_filter(press, sigma=sigma)

# down-sampling per il plot
step = max(1, int(round(100/down_perc)))
press_disp = press[::step, ::step]

# ───────── layout immagini
col1, col2 = st.columns(2)
with col1:
    st.image(img_rgb, caption="Originale", use_column_width=True)
with col2:
    st.image((quant_rgb*255).astype(np.uint8), caption="Quantizzata", use_column_width=True)

# ───────── mappa 2-D
st.subheader("Mappa 2-D pressione [MPa]")
if np.isnan(press_disp).all():
    st.warning("Nessun pixel valido. ↑ Aumenta ΔE* max o abbassa la soglia bianco.")
else:
    z = np.flipud(press_disp)
    zmin, zmax = np.nanmin(z), np.nanmax(z)
    fig = go.Figure(go.Contour(
        z=z, contours_coloring="heatmap",
        colorscale=cmap, zmin=zmin, zmax=zmax,
        showscale=True, colorbar_title="MPa", line_width=0,
        hovertemplate="x=%{x}<br>y=%{y}<br>P=%{z:.2f} MPa<extra></extra>",
    ))
    fig.add_contour(
        z=z, colorscale="Greys", showscale=False,
        line_width=0.4,
        contours=dict(start=zmin, end=zmax, size=0.1)
    )
    fig.update_xaxes(title="x [px]")
    fig.update_yaxes(title="y [px] (origine in basso)")
    fig.update_layout(height=700, width=900, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

# ───────── download
ys, xs = np.indices((h, w))
df = pd.DataFrame({"x_px": xs.ravel(), "y_px": (h-1-ys).ravel(),
                   "Pressione": press.ravel()}).dropna()
st.download_button("CSV pressione", df.to_csv(index=False).encode(),
                   "press_map.csv", "text/csv")
