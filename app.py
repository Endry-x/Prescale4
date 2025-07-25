from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from skimage import color, morphology
from skimage.color import deltaE_ciede2000
from sklearn.cluster import KMeans

# ─────────────── costanti palette default ───────────────────────────
DEFAULT_PAL_PATH = Path(__file__).with_name("palette_default.jpg")
VALUE_START, VALUE_STEP = 0.1, 0.2

# ─────────────── configurazione pagina ─────────────────────────────
st.set_page_config(page_title="Quantizzazione con palette", layout="wide")
st.title("Quantizza l’immagine e analizza i pixel")

# ─────────────── Upload ────────────────────────────────────────────
u1, u2 = st.columns(2)
with u1:
    up_img = st.file_uploader("① Immagine da analizzare", ["png", "jpg", "jpeg"])
with u2:
    up_pal = st.file_uploader("② (opz.) Strip-palette personalizzata", ["png", "jpg", "jpeg"])

if up_img is None:
    st.info("Carica almeno l’immagine da analizzare.")
    st.stop()

img_rgb = Image.open(up_img).convert("RGB")
arr_rgb = np.asarray(img_rgb) / 255.0
h, w, _ = arr_rgb.shape
lab_img = color.rgb2lab(arr_rgb)

# palette
pal_source = up_pal or DEFAULT_PAL_PATH.open("rb")
pal_rgb = Image.open(pal_source).convert("RGB")

# ─────────────── Sidebar parametri ─────────────────────────────────
st.sidebar.header("1) Estrazione palette")
crop_pct = st.sidebar.slider("Ritaglia bordi palette (%)", 0, 40, 10)
n_col    = st.sidebar.number_input("Colori da estrarre", 1, 10, 8)

st.sidebar.header("2) Tolleranza ΔE*")
delta_thr = st.sidebar.number_input("ΔE* max", 0.0, 50.0, 18.0, 0.1)

st.sidebar.header("3) Filtro fondo & rumore")
L_white_min = st.sidebar.slider("Soglia bianco  L min", 80, 100, 95)
min_noise   = st.sidebar.number_input("Area minima rumore (px²)", 1, 5000, 200)

st.sidebar.header("4) Nuvola di punti")
sample_max  = st.sidebar.number_input("Campione max pixel", 1_000, 200_000, 50_000, 1_000)
dot_size    = st.sidebar.slider("Dimensione punto (px)", 2, 12, 4)

# ─────────────── Estrazione colori (k-means) ───────────────────────
pal_arr = np.asarray(pal_rgb) / 255.0
ph = pal_arr.shape[0]
crop = int(ph * crop_pct / 100)
samples = color.rgb2lab(pal_arr[crop:ph - crop])[..., :]

samples = samples[samples[:, :, 0] < 95].reshape(-1, 3)

centers_lab = KMeans(n_clusters=int(n_col), n_init="auto", random_state=0)\
                .fit(samples).cluster_centers_
centers_lab = centers_lab[np.argsort(centers_lab[:, 0])[::-1]]
centers_rgb = color.lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)

values = np.round(np.linspace(VALUE_START,
                              VALUE_START + VALUE_STEP * (len(centers_lab) - 1),
                              len(centers_lab)), 2)

# ─────────────── ΔE*, maschere, quantizzazione ─────────────────────
delta_stack = np.stack(
    [deltaE_ciede2000(lab_img, c.reshape(1, 1, 3)) for c in centers_lab],
    axis=0
)
idx_min, delta_min = delta_stack.argmin(axis=0), delta_stack.min(axis=0)

white_mask = (
    (lab_img[:, :, 0] >= L_white_min) &
    (np.abs(lab_img[:, :, 1]) + np.abs(lab_img[:, :, 2]) < 3)
)
valid = (delta_min <= delta_thr) & ~white_mask
valid = morphology.remove_small_objects(valid, min_size=min_noise)
valid = morphology.binary_closing(valid, morphology.disk(1))

quant_rgb = np.ones_like(arr_rgb)
quant_rgb[valid] = centers_rgb[idx_min[valid]]

val_map = np.full((h, w), np.nan)
for i, v in enumerate(values):
    val_map[(idx_min == i) & valid] = v

# ─────────────── DataFrame pixel con valori ────────────────────────
ys, xs = np.indices((h, w))
df_pix = pd.DataFrame({
    "x": xs.flatten(),
    "y": (h - 1 - ys).flatten(),   # origine in basso
    "Value": val_map.flatten()
}).dropna()

# ─────────────── Mostra immagini ───────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.image(img_rgb, caption="Originale", use_column_width=True)
with c2:
    st.image((quant_rgb * 255).astype(np.uint8), caption="Quantizzata", use_column_width=True)

# ─────────────── Nuvola di punti interattiva (Plotly) ──────────────
st.subheader("Nuvola di punti interattiva (hover → valore quantizzato)")
if df_pix.empty:
    st.info("Nessun pixel valido con i parametri correnti.")
else:
    df_show = df_pix if len(df_pix) <= sample_max else df_pix.sample(sample_max, random_state=0)
    if len(df_pix) > sample_max:
        st.write(f"Mostrati {sample_max:,} pixel su {len(df_pix):,}.")
    fig = px.scatter(
        df_show, x="x", y="y", color="Value",
        color_continuous_scale="Reds",
        height=650, width=900,
        hover_data={"x": True, "y": True, "Value": ":.2f"},
    )
    fig.update_yaxes(autorange="reversed", title="y [px] (origine in basso)")
    fig.update_xaxes(title="x [px]")
    fig.update_traces(marker=dict(size=dot_size))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────── Anteprima & download CSV ──────────────────────────
st.subheader("Anteprima valori per pixel")
st.dataframe(df_pix.head(10_000), use_container_width=True)

st.download_button("📥 CSV valori per pixel",
                   df_pix.to_csv(index=False).encode(),
                   "valori_pixel.csv", "text/csv")

# ─────────────── Download PNG quantizzato ──────────────────────────
buf_img = Image.fromarray((quant_rgb * 255).astype(np.uint8))
st.download_button("📥 PNG quantizzato",
                   buf_img.tobytes(), "quantizzato.png", "image/png")
