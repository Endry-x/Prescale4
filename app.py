import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color
from skimage.color import deltaE_ciede2000
from sklearn.cluster import KMeans

# ------------------- Config pagina -----------------------------------
st.set_page_config(page_title="Quantizzazione con palette", layout="wide")
st.title("Quantizza l’immagine usando la palette di riferimento")

# ------------------- Upload immagini ---------------------------------
col_u1, col_u2 = st.columns(2)
with col_u1:
    up_img = st.file_uploader("① Immagine da analizzare", ["png", "jpg", "jpeg"])
with col_u2:
    up_pal = st.file_uploader("② Strip-palette (8 tacche)", ["png", "jpg", "jpeg"])

if not (up_img and up_pal):
    st.info("Carica entrambe le immagini.")
    st.stop()

img_rgb  = Image.open(up_img).convert("RGB")
arr_rgb  = np.asarray(img_rgb) / 255.0
h, w, _  = arr_rgb.shape
lab_img  = color.rgb2lab(arr_rgb)

# ------------------- Parametri palette -------------------------------
st.sidebar.header("Parametri palette / clustering")
crop_pct = st.sidebar.slider("Ritaglia bordi palette (%)", 0, 40, 10)
n_col    = st.sidebar.number_input("Colori da estrarre", 1, 10, 8)

# estrai pixel utili
pal_arr = np.asarray(Image.open(up_pal).convert("RGB")) / 255.0
ph = pal_arr.shape[0]
crop = int(ph * crop_pct / 100)
samples = color.rgb2lab(pal_arr[crop:ph-crop])       # Lab
samples = samples[samples[:, :, 0] < 95].reshape(-1, 3)

centers_lab = KMeans(n_clusters=int(n_col), n_init="auto", random_state=0)\
                 .fit(samples).cluster_centers_
centers_lab = centers_lab[np.argsort(centers_lab[:, 0])[::-1]]         # chiaro→scuro
centers_rgb = color.lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)

values = np.round(np.linspace(0.1, 0.1 + 0.2*(len(centers_lab)-1),
                              len(centers_lab)), 2)

# ------------------- Parametri ΔE* -----------------------------------
st.sidebar.header("Tolleranza ΔE*")
delta_thr = st.sidebar.number_input("ΔE* max", 0.0, 50.0, 20.0, 0.1)

delta_stack = np.stack(
    [deltaE_ciede2000(lab_img, c.reshape(1, 1, 3)) for c in centers_lab],
    axis=0
)
idx_min   = delta_stack.argmin(axis=0)
delta_min = delta_stack.min(axis=0)
valid     = delta_min <= delta_thr

# ------------------- Img quantizzata & valore per pixel --------------
quant_rgb = np.ones_like(arr_rgb)
quant_rgb[valid] = centers_rgb[idx_min[valid]]

val_map = np.full((h, w), np.nan)
for i, v in enumerate(values):
    val_map[idx_min == i] = v

# ------------------- Tabella bande (come prima) ----------------------
stats = []
total_valid = valid.sum()
for i, (lab_c, rgb_c, v) in enumerate(zip(centers_lab, centers_rgb, values)):
    m = (idx_min == i) & valid
    if m.any():
        area_pct = m.sum()/total_valid*100
        max_de = delta_min[m].max()
    else:
        area_pct, max_de = 0, np.nan
    stats.append({
        "RGB_ref": tuple((rgb_c*255).astype(int)),
        "Value": v,
        "Area_%": round(area_pct, 2),
        "ΔE*_max_px": round(max_de, 2) if not np.isnan(max_de) else np.nan
    })
df_stats = pd.DataFrame(stats)

# ------------------- Tabella pixel -----------------------------------
preview_rows = 10_000
ys, xs = np.indices((h, w))
ys_bottom = (h - 1) - ys
df_pix = pd.DataFrame({
    "x": xs.flatten(),
    "y": ys_bottom.flatten(),
    "Value": val_map.flatten()
}).dropna()

st.subheader("Immagine originale vs quantizzata")
c1, c2 = st.columns(2)
with c1:
    st.image(img_rgb, caption="Originale", use_column_width=True)
with c2:
    st.image((quant_rgb*255).astype(np.uint8), caption="Quantizzata", use_column_width=True)

st.subheader("Statistiche per colore")
st.dataframe(df_stats, use_container_width=True)

st.subheader(f"Valore quantizzato per pixel (anteprima {preview_rows:,} righe)")
st.dataframe(df_pix.head(preview_rows), use_container_width=True)

csv_pix = df_pix.to_csv(index=False).encode()
st.download_button("📥 Scarica CSV valori per pixel",
                   csv_pix, "valori_pixel.csv", "text/csv")

# ------------------- Download immagine -------------------------------
buf_img = Image.fromarray((quant_rgb*255).astype(np.uint8))
st.download_button("📥 Scarica PNG quantizzato",
                   buf_img.tobytes(), "quantizzato.png", "image/png")
