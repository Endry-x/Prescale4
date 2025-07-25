from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from skimage import color, morphology
from skimage.color import deltaE_ciede2000
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────────────────────────────
#  ► PARAMETRI FISSI  ◄
# ─────────────────────────────────────────────────────────────────────
PALETTE_PATH = Path(__file__).with_name("palette_default.jpg")
VALUE_START, VALUE_STEP = 0.1, 0.2

# ----- 1. rette che definiscono A/B/C/D:  y = m*x + q  (y = %RH, x = °C)
#         (ordinate in alto→basso)
LINES = [
    {"name": "AB", "m": -1.4, "q":  92},   # sopra → zona A
    {"name": "BC", "m": -1.4, "q":  72},   # fra AB & BC → zona B
    {"name": "CD", "m": -1.4, "q":  52},   # fra BC & CD → zona C
    # sotto CD → zona D
]

# ----- 2. punti (densità, pressione) tracciati dal grafico per ciascuna zona
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

def zone_from_TRH(T: float, RH: float) -> str:
    """Restituisce 'A','B','C' o 'D' per la coppia (T, RH)."""
    if RH > LINES[0]["m"] * T + LINES[0]["q"]:
        return "A"
    elif RH > LINES[1]["m"] * T + LINES[1]["q"]:
        return "B"
    elif RH > LINES[2]["m"] * T + LINES[2]["q"]:
        return "C"
    else:
        return "D"

def pressure_from_density(density: np.ndarray, zone: str) -> np.ndarray:
    """Interpola la curva della zona per passare da densità a pressione."""
    pts = np.array(CURVE_PTS[zone])        # shape (n, 2)
    dens, press = pts[:, 0], pts[:, 1]
    return np.interp(density, dens, press, left=np.nan, right=np.nan)

# ─────────────────────────────────────────────────────────────────────
#  ► INTERFACCIA STREAMLIT                                            │
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quantizzazione → Pressione", layout="wide")
st.title("Quantizza, calcola la pressione e visualizza i pixel")

# ---------- upload ---------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    up_img = st.file_uploader("① Immagine da analizzare", ["png", "jpg", "jpeg"])
with c2:
    up_pal = st.file_uploader("② (opz.) Strip-palette personalizzata", ["png", "jpg", "jpeg"])

if up_img is None:
    st.info("Carica l’immagine da analizzare.")
    st.stop()

# ---------- input temperatura / umidità ------------------------------
st.sidebar.header("Condizioni ambientali")
T_in = st.sidebar.number_input("Temperatura (°C)", -10.0, 60.0, 25.0, 0.5)
RH_in = st.sidebar.number_input("Umidità relativa (%)", 0.0, 100.0, 60.0, 1.0)
zona   = zone_from_TRH(T_in, RH_in)
st.sidebar.write(f"→ Zona **{zona}**")

# ---------- parametri resto quantizzazione ---------------------------
st.sidebar.header("Parametri quantizzazione (come prima)")
crop_pct = st.sidebar.slider("Ritaglia bordi palette (%)", 0, 40, 10)
n_col    = st.sidebar.number_input("Colori da estrarre", 1, 10, 8)
delta_thr = st.sidebar.number_input("ΔE* max", 0.0, 50.0, 18.0, 0.1)
L_white_min = st.sidebar.slider("Soglia bianco L min", 80, 100, 95)
min_noise   = st.sidebar.number_input("Area minima rumore", 1, 5000, 200)
sample_max  = st.sidebar.number_input("Pixel in grafico", 1_000, 200_000, 50_000, 1_000)
dot_size    = st.sidebar.slider("Dimensione punto", 2, 12, 4)

# ---------- carica immagini -----------------------------------------
img_rgb = Image.open(up_img).convert("RGB")
arr_rgb = np.asarray(img_rgb) / 255.0
h, w, _ = arr_rgb.shape
lab_img = color.rgb2lab(arr_rgb)

pal_source = up_pal or PALETTE_PATH.open("rb")
pal_rgb = Image.open(pal_source).convert("RGB")

# ---------- estrai palette ------------------------------------------
pal_arr = np.asarray(pal_rgb) / 255.0
ph = pal_arr.shape[0]
crop = int(ph * crop_pct / 100)
samples = color.rgb2lab(pal_arr[crop:ph - crop])[..., :]
samples = samples[samples[:, :, 0] < 95].reshape(-1, 3)

centers_lab = KMeans(n_clusters=int(n_col), n_init="auto", random_state=0).fit(samples).cluster_centers_
centers_lab = centers_lab[np.argsort(centers_lab[:, 0])[::-1]]
centers_rgb = color.lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
values = np.round(np.linspace(VALUE_START,
                              VALUE_START + VALUE_STEP * (len(centers_lab) - 1),
                              len(centers_lab)), 2)

# ---------- quantizzazione ------------------------------------------
delta_stack = np.stack([deltaE_ciede2000(lab_img, c.reshape(1, 1, 3))
                        for c in centers_lab], axis=0)
idx_min, delta_min = delta_stack.argmin(axis=0), delta_stack.min(axis=0)

white_mask = ((lab_img[:, :, 0] >= L_white_min) &
              (np.abs(lab_img[:, :, 1]) + np.abs(lab_img[:, :, 2]) < 3))
valid = (delta_min <= delta_thr) & ~white_mask
valid = morphology.remove_small_objects(valid, min_size=min_noise)
valid = morphology.binary_closing(valid, morphology.disk(1))

quant_rgb = np.ones_like(arr_rgb)
quant_rgb[valid] = centers_rgb[idx_min[valid]]

val_map = np.full((h, w), np.nan)
for i, v in enumerate(values):
    val_map[(idx_min == i) & valid] = v

# ---------- calcola pressione ---------------------------------------
pressure_map = pressure_from_density(val_map, zona)

# ---------- DataFrame per grafico -----------------------------------
ys, xs = np.indices((h, w))
df = pd.DataFrame({
    "x_px": xs.flatten(),
    "y_px": (h - 1 - ys).flatten(),
    "Densità": val_map.flatten(),
    "Pressione": pressure_map.flatten(),
}).dropna()

# ---------- immagini -------------------------------------------------
c3, c4 = st.columns(2)
with c3:
    st.image(img_rgb, caption="Originale", use_column_width=True)
with c4:
    st.image((quant_rgb * 255).astype(np.uint8), caption="Quantizzata", use_column_width=True)

# ---------- scatter interattivo Plotly -------------------------------
st.subheader(f"Scatter interattivo – Pressione (zona {zona})")

if df.empty:
    st.info("Nessun pixel valido con i parametri correnti.")
else:
    df_show = df if len(df) <= sample_max else df.sample(sample_max, random_state=0)
    if len(df) > sample_max:
        st.write(f"Mostrati {sample_max:,} punti su {len(df):,}.")
    fig = px.scatter(
        df_show, x="x_px", y="Pressione", color="Densità",
        color_continuous_scale="Reds",
        height=650, width=900,
        hover_data={"x_px": True, "y_px": True,
                    "Pressione": ":.2f", "Densità": ":.2f"},
    )
    fig.update_traces(marker=dict(size=dot_size))
    fig.update_xaxes(title="x [px]")
    fig.update_yaxes(title="Pressione [MPa]")
    st.plotly_chart(fig, use_container_width=True)

# ---------- download -------------------------------------------------
csv = df.to_csv(index=False).encode()
st.download_button("📥 CSV (x_px, y_px, densità, pressione)", csv,
                   "pixel_pressione.csv", "text/csv")
buf_img = Image.fromarray((quant_rgb * 255).astype(np.uint8))
st.download_button("📥 PNG quantizzato", buf_img.tobytes(),
                   "quantizzato.png", "image/png")
