from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color
from skimage.color import deltaE_ciede2000
from sklearn.cluster import KMeans

# ------------ percorso palette di default ----------------------------
DEFAULT_PAL_PATH = Path(__file__).with_name("palette_default.jpg")  # <─ cambia qui se serve

# ------------ pagina --------------------------------------------------
st.set_page_config(page_title="Quantizzazione con palette", layout="wide")
st.title("Quantizza l’immagine usando la palette di riferimento")

# ------------ upload --------------------------------------------------
u1, u2 = st.columns(2)
with u1:
    up_img = st.file_uploader("① Immagine da analizzare", ["png", "jpg", "jpeg"])
with u2:
    up_pal = st.file_uploader("② (opz.) Strip-palette personalizzata", ["png", "jpg", "jpeg"])

if up_img is None:
    st.info("Carica almeno l’immagine da analizzare.")
    st.stop()

#  immagine da analizzare
img_rgb = Image.open(up_img).convert("RGB")
arr_rgb = np.asarray(img_rgb) / 255.0
h, w, _ = arr_rgb.shape
lab_img = color.rgb2lab(arr_rgb)

#  palette (upload oppure default)
if up_pal is None:
    if not DEFAULT_PAL_PATH.exists():
        st.error(f"Palette di default non trovata in {DEFAULT_PAL_PATH}")
        st.stop()
    pal_source = DEFAULT_PAL_PATH.open("rb")
else:
    pal_source = up_pal
pal_rgb = Image.open(pal_source).convert("RGB")

# ------------ parametri palette --------------------------------------
st.sidebar.header("Estrazione palette")
crop_pct = st.sidebar.slider("Ritaglia bordi palette (%)", 0, 40, 10)
n_col    = st.sidebar.number_input("Colori da estrarre", 1, 10, 8)

pal_arr = np.asarray(pal_rgb) / 255.0
ph = pal_arr.shape[0]
crop = int(ph * crop_pct / 100)
pal_crop = pal_arr[crop:ph-crop]

pal_lab = color.rgb2lab(pal_crop)
samples = pal_lab[pal_lab[:, :, 0] < 95].reshape(-1, 3)  # elimina quasi-bianchi

centers_lab = KMeans(n_clusters=int(n_col), n_init="auto", random_state=0).fit(samples).cluster_centers_
centers_lab = centers_lab[np.argsort(centers_lab[:, 0])[::-1]]           # chiaro→scuro
centers_rgb = color.lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)

values = np.round(np.linspace(0.1, 0.1 + 0.2*(len(centers_lab)-1), len(centers_lab)), 2)

# ------------ ΔE* e assegnazione -------------------------------------
st.sidebar.header("Tolleranza ΔE*")
delta_thr = st.sidebar.number_input("ΔE* max", 0.0, 50.0, 20.0, 0.1)

delta_stack = np.stack(
    [deltaE_ciede2000(lab_img, c.reshape(1, 1, 3)) for c in centers_lab],
    axis=0
)
idx_min   = delta_stack.argmin(axis=0)
delta_min = delta_stack.min(axis=0)
valid     = delta_min <= delta_thr

quant_rgb = np.ones_like(arr_rgb)
quant_rgb[valid] = centers_rgb[idx_min[valid]]

val_map = np.full((h, w), np.nan)
for i, v in enumerate(values):
    val_map[idx_min == i] = v

# ------------ dataframe pixel ----------------------------------------
ys, xs = np.indices((h, w))
ys_bottom = (h - 1) - ys
df_pix = pd.DataFrame({
    "x": xs.flatten(),
    "y": ys_bottom.flatten(),
    "Value": val_map.flatten()
}).dropna()

# ------------ immagini a confronto -----------------------------------
c1, c2 = st.columns(2)
with c1:
    st.image(img_rgb, caption="Originale", use_column_width=True)
with c2:
    st.image((quant_rgb*255).astype(np.uint8), caption="Quantizzata", use_column_width=True)

# ------------ nuvola di punti ----------------------------------------
st.sidebar.header("Grafico nuvola")
sample_max = st.sidebar.number_input("Campione max punti", 1000, 100_000, 30_000, 1000)

st.subheader("Nuvola di punti (x-y, colore = valore quantizzato)")
if df_pix.empty:
    st.info("Nessun pixel valido con la soglia ΔE* selezionata.")
else:
    df_show = df_pix if len(df_pix) <= sample_max else df_pix.sample(sample_max, random_state=0)
    if len(df_pix) > sample_max:
        st.write(f"Mostrati {sample_max} pixel su {len(df_pix):,}.")
    fig_sc, ax_sc = plt.subplots()
    sc = ax_sc.scatter(df_show["x"], df_show["y"], c=df_show["Value"],
                       cmap="Reds", s=4)
    ax_sc.set_xlabel("x [px]")
    ax_sc.set_ylabel("y [px] (origine in basso)")   # niente invert_yaxis()
    fig_sc.colorbar(sc, ax=ax_sc, label="Valore quantizzato")
    st.pyplot(fig_sc)

# ------------ tabella anteprima & download ---------------------------
st.subheader("Anteprima valori per pixel")
st.dataframe(df_pix.head(10_000), use_container_width=True)

st.download_button("📥 CSV valori per pixel",
                   df_pix.to_csv(index=False).encode(),
                   "valori_pixel.csv", "text/csv")

# ------------ download immagine quantizzata --------------------------
buf_img = Image.fromarray((quant_rgb*255).astype(np.uint8))
st.download_button("📥 PNG quantizzato",
                   buf_img.tobytes(), "quantizzato.png", "image/png")
