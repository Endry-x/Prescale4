# -------------- (… tutto il codice sopra invariato fino a press_map ...) -----

# ─── opzionale: gaussian blur per ammorbidire il campo ─────────────
import importlib
sigma = st.sidebar.slider("Sfumatura (σ px)", 0, 5, 1,
                          help="0 = nessuna sfumatura. "
                               "Richiede il pacchetto SciPy.")
if sigma > 0 and importlib.util.find_spec("scipy") is not None:
    from scipy.ndimage import gaussian_filter
    press_smooth = gaussian_filter(press_map, sigma=sigma)
else:
    press_smooth = press_map

# ─── down-sampling per la sola visualizzazione ─────────────────────
step = max(1, int(round(100 / down_perc)))
press_show = press_smooth[::step, ::step]

# ─── mappa 2-D con Contour filled ──────────────────────────────────
st.subheader("Campo di pressione – filled contour")

if np.isnan(press_show).all():
    st.info("Nessun pixel valido con i parametri scelti.")
else:
    import plotly.graph_objects as go

    z = np.flipud(press_show)  # origine in basso
    # Definiamo una colorscale trasparente per i NaN
    zmin, zmax = np.nanmin(z), np.nanmax(z)
    fig = go.Figure(
        go.Contour(
            z=z,
            colorscale=cmap,
            zmin=zmin, zmax=zmax,
            contours_coloring="heatmap",
            showscale=True,
            colorbar_title="MPa",
            hovertemplate="x=%{x}<br>y=%{y}<br>P=%{z:.2f} MPa<extra></extra>",
        )
    )

    # Rimuovi contorno nero esterno
    fig.update_traces(line_width=0)
    fig.update_yaxes(title="y [px] (origine in basso)")
    fig.update_xaxes(title="x [px]")

    fig.update_layout(
        height=700, width=900,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------- download CSV invariato -----------------------------
ys, xs = np.indices((h, w))
df = pd.DataFrame({
    "x_px": xs.flatten(),
    "y_px": (h - 1 - ys).flatten(),
    "Pressione": press_map.flatten()
}).dropna()
st.download_button("📥 CSV pressione", df.to_csv(index=False).encode(),
                   "press_map.csv", "text/csv")
