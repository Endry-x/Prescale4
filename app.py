from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from skimage import color, morphology
from skimage.color import deltaE_ciede2000
from sklearn.cluster import KMeans
import importlib

# ───────── costanti
PALETTE_PATH = Path(__file__).with_name("palette_default.jpg")
VALUE_START, VALUE_STEP = 0.1, 0.2
LINES = [{"m":-1.4,"q":92},{"m":-1.4,"q":72},{"m":-1.4,"q":52}]
CURVE_PTS = {
    "A":[(0.2,0.4),(0.3,0.55),(0.5,0.8),(0.8,1.25),(1.0,1.55),(1.2,1.95),(1.4,2.4),(1.5,2.65)],
    "B":[(0.2,0.35),(0.3,0.5),(0.5,0.75),(0.8,1.2),(1.0,1.45),(1.2,1.8),(1.4,2.2),(1.5,2.45)],
    "C":[(0.2,0.3),(0.3,0.45),(0.5,0.7),(0.8,1.05),(1.0,1.3),(1.2,1.65),(1.4,2.0),(1.5,2.25)],
    "D":[(0.2,0.25),(0.3,0.4),(0.5,0.6),(0.8,0.95),(1.0,1.2),(1.2,1.5),(1.4,1.85),(1.5,2.1)],
}

# ───────── helper
def zone_from_TRH(T, RH):
    if RH > LINES[0]["m"]*T+LINES[0]["q"]: return "A"
    if RH > LINES[1]["m"]*T+LINES[1]["q"]: return "B"
    if RH > LINES[2]["m"]*T+LINES[2]["q"]: return "C"
    return "D"
def pressure_from_density(d,z):
    pts=np.array(CURVE_PTS[z]);return np.interp(d,pts[:,0],pts[:,1],np.nan,np.nan)

# ───────── UI
st.set_page_config(page_title="Mappa pressione 2-D", layout="wide")
st.title("Campo di pressione [MPa] sui pixel")

up_img=st.file_uploader("Immagine da analizzare",["png","jpg","jpeg"])
up_pal=st.file_uploader("Palette (opz.)",["png","jpg","jpeg"])
if up_img is None: st.stop()

# sidebar
st.sidebar.header("Condizioni")
T=st.sidebar.number_input("Temperatura (°C)",-10.,60.,25.)
RH=st.sidebar.number_input("Umidità relativa (%)",0.,100.,60.)
zona=zone_from_TRH(T,RH);st.sidebar.write(f"→ Zona **{zona}**")

st.sidebar.header("Quantizzazione")
crop_pct=st.sidebar.slider("Crop palette %",0,40,10)
k=st.sidebar.number_input("K-means cluster",1,10,8)
delta_thr=st.sidebar.slider("ΔE* max",0.,50.,25.)
L_white=st.sidebar.slider("L soglia bianco",80,100,95)
min_noise=st.sidebar.number_input("Area min rumore",1,5000,200)

st.sidebar.header("Display")
down_perc=st.sidebar.slider("Riduci risoluzione %",10,100,60)
cmap=st.sidebar.selectbox("Colormap",["Turbo","Jet","Viridis","RdYlBu"],0)
sigma=st.sidebar.slider("Blur σ px",0,5,1)

# ───────── immagini
rgb=np.asarray(Image.open(up_img).convert("RGB"))/255.
h,w,_=rgb.shape; lab=color.rgb2lab(rgb)

pal=np.asarray(Image.open(up_pal or PALETTE_PATH).convert("RGB"))/255.
crop=int(pal.shape[0]*crop_pct/100)
samp=color.rgb2lab(pal[crop:-crop]); samp=samp[samp[:,:,0]<95].reshape(-1,3)

cent_lab=KMeans(k, n_init="auto", random_state=0).fit(samp).cluster_centers_
cent_lab=cent_lab[np.argsort(cent_lab[:,0])[::-1]]
cent_rgb=color.lab2rgb(cent_lab.reshape(1,-1,3)).reshape(-1,3)
values=np.round(np.linspace(VALUE_START,VALUE_START+VALUE_STEP*(k-1),k),2)

# ΔE* e maschere
delta=np.stack([deltaE_ciede2000(lab,c.reshape(1,1,3)) for c in cent_lab])
idx,dmin=delta.argmin(0),delta.min(0)
white=((lab[:,:,0]>=L_white)&(np.abs(lab[:,:,1])+np.abs(lab[:,:,2])<3))
mask=(dmin<=delta_thr)&~white
mask=morphology.remove_small_objects(mask,min_noise)
mask=morphology.binary_closing(mask,morphology.disk(1))

dens=np.full((h,w),np.nan)
for i,v in enumerate(values): dens[(idx==i)&mask]=v
press=pressure_from_density(dens,zona)

# immagine quantizzata
quant=rgb.copy()
for i,col in enumerate(cent_rgb): quant[(idx==i)&mask]=col

# blur opzionale
if sigma>0 and importlib.util.find_spec("scipy"):
    from scipy.ndimage import gaussian_filter
    press=gaussian_filter(press,sigma=sigma)

# conteggio pixel validi
valid_px=np.count_nonzero(~np.isnan(press))
st.write(f"Pixel validi: **{valid_px} / {h*w}**")

# down-sampling con fallback
step=max(1,int(round(100/down_perc)))
press_disp=press[::step,::step]
if np.count_nonzero(~np.isnan(press_disp))<50:
    step=1; press_disp=press  # usa piena risoluzione

# layout immagini
c1,c2=st.columns(2)
with c1: st.image((rgb*255).astype(np.uint8),"Originale",use_column_width=True)
with c2: st.image((quant*255).astype(np.uint8),"Quantizzata",use_column_width=True)

# mappa 2-D
st.subheader("Mappa 2-D pressione")
if np.isnan(press_disp).all():
    st.error("Ancora nessun pixel valido. Prova ad aumentare ΔE* o ridurre L bianco.")
else:
    z=np.flipud(press_disp); zmin,zmax=np.nanmin(z),np.nanmax(z)
    if np.count_nonzero(~np.isnan(z))<30:          # backup heat-map
        fig=px.imshow(z,origin="lower",color_continuous_scale=cmap,
                       labels=dict(color="MPa"),aspect="auto",
                       height=600,width=900)
    else:
        fig=go.Figure(go.Contour(
            z=z,colorscale=cmap,zmin=zmin,zmax=zmax,
            contours_coloring="heatmap",showscale=True,line_width=0,
            colorbar_title="MPa",
            hovertemplate="x=%{x}<br>y=%{y}<br>P=%{z:.2f} MPa<extra></extra>"))
        fig.add_contour(z=z,colorscale="Greys",showscale=False,
                        line_width=0.4,
                        contours=dict(start=zmin,end=zmax,size=0.1))
    fig.update_xaxes(title="x [px]"); fig.update_yaxes(title="y [px] (origine in basso)")
    fig.update_layout(height=700,width=900,margin=dict(l=40,r=40,t=40,b=40))
    st.plotly_chart(fig,use_container_width=True)

# download CSV
ys,xs=np.indices((h,w))
df=pd.DataFrame({"x_px":xs.ravel(),"y_px":(h-1-ys).ravel(),
                 "Pressione":press.ravel()}).dropna()
st.download_button("CSV pressione",df.to_csv(index=False).encode(),
                   "press_map.csv","text/csv")
