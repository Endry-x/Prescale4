from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from skimage import color, morphology
from skimage.color import deltaE_ciede2000
from sklearn.cluster import KMeans

# ───────────────────── costanti ─────────────────────
PALETTE_PATH = Path(__file__).with_name("palette_default.jpg")
VALUE_START, VALUE_STEP = 0.1, 0.2

LINES = [   # y = m·x + q  (limiti A/B/C/D)
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

# ───────────────────── helper ───────────────────────
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
    dens, press = pts[:, 0], pts[:, 1]
    return np.interp(d, dens, press, left=np.nan, right=np.nan)

# ───────────────────── UI ───────────────────────────
st.set_page_config(page_title="Pixel → Pressione", layout="wide")
st.title("Quantizza immagine e calcola la pressione [MPa] per pixel")

# upload
col_u1, col_u2 = st.columns(2)
with col_u1:
    up_img = st.file_uploader("① Immagine bande", ["png", "jpg", "jpeg"])
with col_u2:
    up_pal = st.file_uploader("② Palette (opz.)", ["png", "jpg", "jpeg"])

if up_img is None:
    st.stop()

# condizioni ambiente
st.sidebar.header("Condizioni ambiente")
T_in = st.sidebar.number_input("Temperatura (°C)", -10.0, 60.0, 25.0, 0.5)
RH_in = st.sidebar.number_input("Umidità relativa (%)", 0.0, 100.0, 60.0, 1.0)
zone = zone_from_TRH(T_in, RH_in)
st.sidebar.write(f"→ Zona **{zone}**")

# parametri quantizzazione
st.sidebar.header("Parametri quantizzazione")
crop_pct  = st.sidebar.slider("Ritaglia palette (%)", 0, 40, 10)
n_col     = st.sidebar.number_input("Colori da estrarre", 1, 10, 8)
delta_thr = st.sidebar.number_input("ΔE* max", 0.0, 50.0, 18.0, 0.1)
L_white   = st.sidebar.slider("Soglia bianco L min", 80, 100, 95)
min_noise = st.sidebar.number_input("Area minima rumore", 1, 5000, 200)

st.sidebar.header("Plot")
sample_max = st.sidebar.number_input("Pixel max", 1_000, 200_000, 60_000, 1_000)
dot_size   = st.sidebar.slider("Dimensione punto", 2, 12, 4)

# ───────── carica immagini & palette ─────────
img_rgb = Image.open(up_img).convert("RGB")
arr_rgb = np.asarray(img_rgb)/255.0
h, w, _ = arr_rgb.shape
lab_img = color.rgb2lab(arr_rgb)

pal_src = up_pal or PALETTE_PATH.open("rb")
pal_rgb = Image.open(pal_src).convert("RGB")
pal_arr = np.asarray(pal_rgb)/255.0

# estrai palette
ph = pal_arr.shape[0]
crop = int(ph*crop_pct/100)
samples = color.rgb2lab(pal_arr[crop:ph-crop])
samples = samples[samples[:,:,0] < 95].reshape(-1,3)

cent_lab = KMeans(n_clusters=int(n_col), n_init="auto", random_state=0).fit(samples).cluster_centers_
cent_lab = cent_lab[np.argsort(cent_lab[:,0])[::-1]]
cent_rgb = color.lab2rgb(cent_lab.reshape(1,-1,3)).reshape(-1,3)
values   = np.round(np.linspace(VALUE_START,
                                VALUE_START+VALUE_STEP*(len(cent_lab)-1),
                                len(cent_lab)), 2)

# ΔE*
delta = np.stack([deltaE_ciede2000(lab_img, c.reshape(1,1,3)) for c in cent_lab], axis=0)
idx_min, delta_min = delta.argmin(axis=0), delta.min(axis=0)

white = ((lab_img[:,:,0] >= L_white) &
         (np.abs(lab_img[:,:,1]) + np.abs(lab_img[:,:,2]) < 3))
valid = (delta_min <= delta_thr) & ~white
valid = morphology.remove_small_objects(valid, min_size=min_noise)
valid = morphology.binary_closing(valid, morphology.disk(1))

quant_rgb = np.ones_like(arr_rgb)
quant_rgb[valid] = cent_rgb[idx_min[valid]]

val_map = np.full((h,w), np.nan)
for i, v in enumerate(values):
    val_map[(idx_min == i)&valid] = v

press_map = pressure_from_density(val_map, zone)

# dataframe
ys, xs = np.indices((h,w))
df = pd.DataFrame({
    "x_px": xs.flatten(),
    "y_px": (h-1-ys).flatten(),
    "Densità": val_map.flatten(),
    "Pressione": press_map.flatten()
}).dropna()

# immagini
c1, c2 = st.columns(2)
with c1:
    st.image(img_rgb, caption="Originale", use_column_width=True)
with c2:
    st.image((quant_rgb*255).astype(np.uint8), caption="Quantizzata", use_column_width=True)

# tabs
tab2d, tab3d = st.tabs(["💠 Mappa 2-D", "🧊 Scatter 3-D"])

with tab2d:
    st.subheader("Mappa 2-D: pressione continua")
    if np.isnan(press_map).all():
        st.info("Nessun pixel valido.")
    else:
        # opzionale: filtra/sfuma leggermente solo per visualizzare
        # from scipy.ndimage import gaussian_filter
        # z_show = gaussian_filter(press_map, sigma=1)
        z_show = press_map

        fig2d = px.imshow(
            np.flipud(z_show),        # origine in basso
            origin="lower",
            color_continuous_scale="Jet",   # Jet, Turbo, Viridis…
            aspect="auto",
            labels=dict(color="Pressione [MPa]"),
            height=700, width=900
        )

        # isolinee sottili (facoltative)
        fig2d.add_contour(
            z=np.flipud(z_show),
            colorscale="Greys",
            showscale=False,
            line_width=0.5,
            contours=dict(showlabels=False, start=np.nanmin(z_show),
                          end=np.nanmax(z_show), size=0.1)
        )

        st.plotly_chart(fig2d, use_container_width=True)


with tab3d:
    st.subheader("Scatter 3-D: z = pressione [MPa]")
    if df.empty:
        st.info("Nessun pixel valido.")
    else:
        df_show = df if len(df)<=sample_max else df.sample(sample_max, random_state=0)
        fig3d = px.scatter_3d(
            df_show, x="x_px", y="y_px", z="Pressione",
            color="Pressione", color_continuous_scale="Turbo",
            height=700, width=900,
            hover_data={"x_px":True,"y_px":True,"Pressione":":.2f","Densità":":.2f"},
        )
        fig3d.update_traces(marker=dict(size=dot_size))
        fig3d.update_layout(scene=dict(
            xaxis_title="x [px]",
            yaxis_title="y [px] (origine in basso)",
            zaxis_title="Pressione [MPa]"
        ))
        st.plotly_chart(fig3d, use_container_width=True)

# download
st.download_button("📥 CSV (x_px, y_px, densità, pressione)",
                   df.to_csv(index=False).encode(),
                   "pixel_pressione.csv", "text/csv")
buf = Image.fromarray((quant_rgb*255).astype(np.uint8))
st.download_button("📥 PNG quantizzato", buf.tobytes(),
                   "quantizzato.png", "image/png")
