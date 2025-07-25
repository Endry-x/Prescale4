import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color
from skimage.color import deltaE_ciede2000
from sklearn.cluster import KMeans

# ----------------------- Config pagina ---------------------------------
st.set_page_config(page_title="Quantizzazione con palette 8 rossi", layout="wide")
st.title("Quantizza l’immagine usando la palette fornita (8 colori)")

# ----------------------- Upload immagini -------------------------------
c_up1, c_up2 = st.columns(2)
with c_up1:
    up_img = st.file_uploader("① Immagine da analizzare", ["png", "jpg", "jpeg"])
with c_up2:
    up_pal = st.file_uploader("② Strip-palette (8 tacche rosse)", ["png", "jpg", "jpeg"])

if not (up_img and up_pal):
    st.info("Carica entrambe le immagini per procedere.")
    st.stop()

img_rgb = Image.open(up_img).convert("RGB")
arr_rgb = np.asarray(img_rgb) / 255.0
h, w, _ = arr_rgb.shape
lab_img = color.rgb2lab(arr_rgb)

# ----------------------- Estrazione colori palette ---------------------
st.sidebar.header("Parametri estrazione palette")
crop_pct = st.sidebar.slider("Ritaglia bordi palette (%)", 0, 40, 10)
n_col_sl = st.sidebar.slider("Colori da estrarre (slider)", 1, 10, 8)
n_col_in = st.sidebar.number_input("Colori da estrarre (num.)", 1, 10, n_col_sl)
n_colors = int(n_col_in)

pal_rgb = Image.open(up_pal).convert("RGB")
pw, ph = pal_rgb.size
crop = int(ph * crop_pct / 100)
pal_crop = pal_rgb.crop((0, crop, pw, ph - crop))
pal_arr = np.asarray(pal_crop) / 255.0
pal_lab = color.rgb2lab(pal_arr)

# elimina quasi-bianchi
mask_pal = pal_lab[:, :, 0] < 95
samples = pal_lab[mask_pal].reshape(-1, 3)

kmeans = KMeans(n_clusters=n_colors, n_init="auto", random_state=0)
centers_lab = kmeans.fit(samples).cluster_centers_
# ordina per L decrescente ⇒ più chiaro prima
centers_lab = centers_lab[np.argsort(centers_lab[:, 0])[::-1]]  # L alto = chiaro

# ----------------------- mappa valori 0.1 → 1.5 ------------------------
values = np.round(np.linspace(0.1, 0.1 + 0.2 * (n_colors - 1), n_colors), 2)
# palette RGB per la nuova immagine
centers_rgb = color.lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)

# ----------------------- ΔE* e assegnazione ----------------------------
st.sidebar.header("Tolleranza assegnazione")
delta_thr_sl = st.sidebar.slider("ΔE* max (slider)", 0.0, 50.0, 20.0, 0.1)
delta_thr_in = st.sidebar.number_input("ΔE* max (num.)", 0.0, 50.0, delta_thr_sl, 0.1)
delta_thr = float(delta_thr_in)

delta_stack = np.stack(
    [deltaE_ciede2000(lab_img, c.reshape(1, 1, 3)) for c in centers_lab],
    axis=0
)                             # shape (k, H, W)

idx_min = delta_stack.argmin(axis=0)         # indici colore più vicino
delta_min = delta_stack.min(axis=0)          # ΔE* minimo

# maschera valida
valid = delta_min <= delta_thr

# immagine quantizzata
quant_rgb = np.ones_like(arr_rgb)            # init bianco
quant_rgb[valid] = centers_rgb[idx_min[valid]]

# mappa valori numerici
val_map = np.full((h, w), np.nan)
for i in range(n_colors):
    val_map[idx_min == i] = values[i]

# ----------------------- Statistiche per colore ------------------------
stats = []
total_valid = valid.sum()
for i in range(n_colors):
    mask_i = (idx_min == i) & valid
    if not mask_i.any():
        area_pct = 0
        max_de = np.nan
    else:
        area_pct = mask_i.sum() / total_valid * 100
        max_de = delta_min[mask_i].max()
    stats.append({
        "RGB_ref": tuple((centers_rgb[i] * 255).astype(int)),
        "Value": values[i],
        "Area_%": round(area_pct, 2),
        "ΔE*_max_px": round(max_de, 2) if not np.isnan(max_de) else np.nan
    })
df_stats = pd.DataFrame(stats)

# ----------------------- Visualizzazione ------------------------------
col_v1, col_v2 = st.columns(2)
with col_v1:
    st.image(img_rgb, caption="Originale", use_column_width=True)
with col_v2:
    st.image((quant_rgb * 255).astype(np.uint8), caption="Quantizzata", use_column_width=True)

st.subheader("Statistiche per colore della palette")
st.dataframe(df_stats, use_container_width=True)

# ----------------------- Download immagine ----------------------------
buf_img = Image.fromarray((quant_rgb * 255).astype(np.uint8))
st.download_button("📥 Scarica PNG quantizzato",
                   data=buf_img.tobytes(),
                   file_name="quantizzato.png",
                   mime="image/png")
