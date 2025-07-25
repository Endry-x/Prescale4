from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color, morphology
from skimage.color import deltaE_ciede2000
from sklearn.cluster import KMeans

# ──────────────────────────────── Costanti ───────────────────────────
DEFAULT_PAL_PATH = Path(__file__).with_name("palette_default.jpg")  # palette interna
VALUE_START = 0.1          # valore assegnato al colore più chiaro
VALUE_STEP  = 0.2          # incremento fra un colore e il successivo

# ───────────────────────────── Config pagina ─────────────────────────
st.set_page_config(page_title="Quantizzazione con palette", layout="wide")
st.title("Quantizza l’immagine usando la palette di riferimento")

# ───────────────────────────── Upload file ───────────────────────────
u1, u2 = st.columns(2)
with u1:
    up_img = st.file_uploader("① Immagine da analizzare", ["png", "jpg", "jpeg"])
with u2:
    up_pal = st.file_uploader("② (opz.) Strip-palette personalizzata", ["png", "jpg", "jpeg"])

if up_img is None:
    st.info("Carica almeno l’immagine da analizzare.")
    st.stop()

# immagine da analizzare
img_rgb  = Image.open(up_img).convert("RGB")
arr_rgb  = np.asarray(img_rgb) / 255.0
h, w, _  = arr_rgb.shape
lab_img  = color.rgb2lab(arr_rgb)

# palette (upload o default)
if up_pal is None:
    if not DEFAULT_PAL_PATH.exists():
        st.error(f"Palette di default non trovata: {DEFAULT_PAL_PATH}")
        st.stop()
    pal_source = DEFAULT_PAL_PATH.open("rb")
else:
    pal_source = up_pal
pal_rgb = Image.open(pal_source).convert("RGB")

# ───────────────────────────── Sidebar parametri ─────────────────────
st.sidebar.header("1) Estrazione palette")
crop_pct = st.sidebar.slider("Ritaglia bordi palette (%)", 0, 40, 10)
n_col    = st.sidebar.number_input("Colori da estrarre", 1, 10, 8)

st.sidebar.header("2) Tolleranza ΔE*")
delta_thr = st.sidebar.number_input("ΔE* max", 0.0, 50.0, 20.0, 0.1)

st.sidebar.header("3) Filtro fondo & rumore")
L_white_min = st.sidebar.slider("Soglia bianco  L min", 80, 100, 94)
min_noise   = st.sidebar.number_input("Area minima rumore (px²)", 1, 5000, 150)

st.sidebar.header("4) Nuvola di punti")
sample_max = st.sidebar.number_input("Campione max punti", 1_000, 100_000, 30_000, 1_000)

# ───────────────────────────── Estrai colori palette ─────────────────
pal_arr = np.asarray(pal_rgb) / 255.0
ph = pal_arr.shape[0]
crop = int(ph * crop_pct / 100)
pal_crop = pal_arr[crop:ph - crop]

pal_lab = color.rgb2lab(pal_crop)
samples = pal_lab[pal_lab[:, :, 0] < 95].reshape(-1, 3)     # scarta quasi-bianchi

centers_lab = KMeans(n_clusters=int(n_col), n_init="auto", random_state=0)\
                .fit(samples).cluster_centers_
centers_lab = centers_lab[np.argsort(centers_lab[:, 0])[::-1]]      # chiaro → scuro
centers_rgb = color.lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)

values = np.round(np.linspace(VALUE_START,
                              VALUE_START + VALUE_STEP * (len(centers_lab) - 1),
                              len(centers_lab)), 2)

# ───────────────────────────── ΔE* e assegnazione ────────────────────
delta_stack = np.stack(
    [deltaE_ciede2000(lab_img, c.reshape(1, 1, 3)) for c in centers_lab],
    axis=0
)
idx_min   = delta_stack.argmin(axis=0)
delta_min = delta_stack.min(axis=0)

# maschera bianco
white_mask = (
    (lab_img[:, :, 0] >= L_white_min) &
    (np.abs(lab_img[:, :, 1]) + np.abs(lab_img[:, :, 2]) < 3)
)

# pixel candidati (validi) = distanza OK e non bianco
valid = (delta_min <= delta_thr) & ~white_mask

# rimuovi piccole macchie & chiudi buchi
valid = morphology.remove_small_objects(valid, min_size=min_noise)
valid = morphology.binary_closing(valid, morphology.disk(1))

# nuova immagine quantizzata
quant_rgb = np.ones_like(arr_rgb)           # fondo bianco
quant_rgb[valid] = centers_rgb[idx_min[valid]]

# mappa valori numerici
val_map = np.full((h, w), np.nan)
for i, v in enumerate(values):
    val_map[(idx_min == i) & valid] = v

# ───────────────────────────── DataFrame pixel ───────────────────────
ys, xs = np.indices((h, w))
ys_bottom = (h - 1) - ys
df_pix = pd.DataFrame({
    "x": xs.flatten(),
    "y": ys_bottom.flatten(),
    "Value": val_map.flatten()
}).dropna()

# ───────────────────────────── Immagini a confronto ──────────────────
c1, c2 = st.columns(2)
with c1:
    st.image(img_rgb, caption="Originale", use_column_width=True)
with c2:
    st.image((quant_rgb * 255).astype(np.uint8), caption="Quantizzata", use_column_width=True)

# ───────────────────────────── Nuvola di punti ───────────────────────
st.subheader("Nuvola di punti (x-y, colore = valore quantizzato)")
if df_pix.empty:
    st.info("Nessun pixel valido con i parametri correnti.")
else:
    df_show = df_pix if len(df_pix) <= sample_max else df_pix.sample(sample_max, random_state=0)
    if len(df_pix) > sample_max:
        st.write(f"Mostrati {sample_max:,} pixel su {len(df_pix):,}.")
    fig_sc, ax_sc = plt.subplots()
    sc = ax_sc.scatter(df_show["x"], df_show["y"],
                       c=df_show["Value"], cmap="Reds", s=4)
    ax_sc.set_xlabel("x [px]")
    ax_sc.set_ylabel("y [px] (origine in basso)")  # non invertiamo più l’asse
    fig_sc.colorbar(sc, ax=ax_sc, label="Valore quantizzato")
    st.pyplot(fig_sc)

# ───────────────────────────── Anteprima & download CSV ──────────────
st.subheader("Anteprima valori per pixel")
st.dataframe(df_pix.head(10_000), use_container_width=True)

st.download_button("📥 Scarica CSV valori per pixel",
                   df_pix.to_csv(index=False).encode(),
                   "valori_pixel.csv", "text/csv")

# ───────────────────────────── Download PNG quantizzato ──────────────
buf_img = Image.fromarray((quant_rgb * 255).astype(np.uint8))
st.download_button("📥 Scarica PNG quantizzato",
                   buf_img.tobytes(), "quantizzato.png", "image/png")
