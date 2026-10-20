import streamlit as st
import pandas as pd
import folium
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
from pathlib import Path

st.set_page_config(
    page_title="Kullerden Yesile",
    page_icon="🌲",
    layout="wide"
)

ROOT = Path(__file__).parent

@st.cache_data
def load_data():
    points = pd.read_csv(
        ROOT / "Manavgat_ML_Dataset_2018_2026.csv"
    )
    preds = pd.read_csv(
        ROOT / "Manavgat_Mekansal_Tahminler.csv"
    )
    return points, preds

points, preds = load_data()
points["id"] = points["system:index"].astype(int)
preds["id"] = preds["id"].astype(int)

st.title("🌲 Küllerden Yeşile")
st.subheader("Manavgat Orman İyileşmesi Dijital İkizi")
st.caption(
    "2021 yangını sonrası uydu gözlemleri ve "
    "Random Forest tahminlerinin pilot arayüzü"
)

# SOL MENU
st.sidebar.header("Harita Kontrolleri")

year = st.sidebar.slider(
    "İncelenecek yıl",
    min_value=2021,
    max_value=2026,
    value=2026
)

selected = st.sidebar.multiselect(
    "Yangın etki sınıfı",
    [2, 3, 4],
    default=[2, 3, 4],
    format_func=lambda x: {
        2: "Düşük",
        3: "Orta",
        4: "Yüksek"
    }[x]
)

show_pred = st.sidebar.checkbox(
    "2026 test tahminlerini göster",
    value=False
)

if not selected:
    st.warning("En az bir sınıf seçin.")
    st.stop()

filtered = points[
    points["etki_sinifi"].isin(selected)
].copy()

# OZET GOSTERGELER
average = filtered[f"NDVI_{year}"].mean()
base2021 = filtered["NDVI_2021"].mean()

c1, c2, c3 = st.columns(3)

c1.metric("Seçilen yıl", str(year))
c2.metric("Ortalama NDVI", f"{average:.4f}")
c3.metric(
    "2021'e göre NDVI farkı",
    f"{average - base2021:+.4f}"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Harita",
    "📈 İyileşme",
    "🤖 Model",
    "🌦️ İklim"
])

# HARITA
with tab1:
    m = folium.Map(
        location=[
            filtered["enlem"].mean(),
            filtered["boylam"].mean()
        ],
        zoom_start=11,
        tiles=None
    )

    try:
        carto_key = st.secrets.get(
            "CARTO_API_KEY", ""
        )
    except FileNotFoundError:
        carto_key = ""

    if carto_key:
        folium.TileLayer(
            tiles=(
                "https://basemaps.cartocdn.com/"
                "rastertiles/light_all/"
                "{z}/{x}/{y}.png"
                f"?key={carto_key}"
            ),
            attr="© OpenStreetMap contributors, © CARTO",
            name="CARTO Positron"
        ).add_to(m)
    else:
        st.info("CARTO anahtarı eklenmedi. Altlık harita boş.")

    scale = LinearColormap(
        [
            "#8c510a",
            "#d8b365",
            "#f6e8c3",
            "#7fbf7b",
            "#006837"
        ],
        vmin=0,
        vmax=0.85,
        caption="NDVI"
    )

    for _, row in filtered.iterrows():
        value = float(row[f"NDVI_{year}"])
        color = scale(value)

        
        folium.CircleMarker(
            location=[
                row["enlem"],
                row["boylam"]
            ],
            radius=9,
            color="#7c3aed",
            weight=3,
            fill=True,
            fill_color="#7c3aed",
            fill_opacity=0.25,
            tooltip="Random Forest tahmini",
            popup=folium.Popup(
                f"<b>2026 Random Forest</b><br>"
                f"Gercek NDVI: {row['NDVI_hedef']:.4f}<br>"
                f"Tahmin NDVI: {row['NDVI_tahmin']:.4f}<br>"
                f"Mutlak hata: {row['mutlak_hata']:.4f}<br>"
                f"Etki sinifi: {int(row['sinif'])}",
                max_width=260
            )
        ).add_to(m)


    if show_pred and year == 2026:
        subset = preds[
            preds["sinif"].isin(selected)
        ].merge(
            points[["id", "enlem", "boylam"]],
            on="id",
            validate="one_to_one"
        )

        for _, row in subset.iterrows():
            folium.CircleMarker(
                [row["enlem"], row["boylam"]],
                radius=8,
                color="#7c3aed",
                weight=2,
                fill=False,
                popup=(
                    f"Gerçek: {row['NDVI_hedef']:.4f}<br>"
                    f"Tahmin: {row['NDVI_tahmin']:.4f}<br>"
                    f"Hata: {row['mutlak_hata']:.4f}"
                )
            ).add_to(m)


    # NDVI renk skalasini haritaya ekle
    scale.add_to(m)

    # Haritayi ornek noktalarina sigdir
    m.fit_bounds([
        [filtered["enlem"].min(), filtered["boylam"].min()],
        [filtered["enlem"].max(), filtered["boylam"].max()]
    ])

    # Haritayi Streamlit'te goster
    st_folium(
        m,
        height=550,
        use_container_width=True,
        returned_objects=[]
    )

    st.caption(
        "Noktalar Sentinel-2 orneklerini temsil eder; "
        "kesintisiz uydu raster haritasi degildir."
    )


# IYILESME
with tab2:
    trend = pd.DataFrame({
        "Yıl": list(range(2018, 2027)),
        "Ortalama NDVI": [
            filtered[f"NDVI_{y}"].mean()
            for y in range(2018, 2027)
        ]
    }).set_index("Yıl")

trend.index = trend.index.astype(str)
st.line_chart(trend)
st.dataframe(trend)

# MODEL
with tab3:
    st.subheader("2026 Mekânsal Test")

    errors = abs(
        preds["NDVI_tahmin"] -
        preds["NDVI_hedef"]
    )

    baseline = abs(
        points.set_index("id").loc[
            preds["id"], "NDVI_2025"
        ].to_numpy() -
        preds["NDVI_hedef"].to_numpy()
    )

    a, b = st.columns(2)
    a.metric("Random Forest MAE", f"{errors.mean():.4f}")
    b.metric("Önceki yıl MAE", f"{baseline.mean():.4f}")

    st.dataframe(preds.head(20))

# IKLIM
with tab4:
    climate_path = ROOT / "Manavgat_Iklim_2022_2026.csv"

    if climate_path.exists():
        climate = pd.read_csv(climate_path)
        climate = climate.set_index("yil")
        climate.index = climate.index.astype(str)
        st.subheader("Ortalama Sıcaklık (°C)")
        st.line_chart(climate["sicaklik_C"])

        st.subheader("Toplam Yağış (mm)")
        st.bar_chart(climate["yagis_mm"])
    else:
        st.info("İklim CSV dosyası henüz eklenmedi.")

st.caption(
    "Araştırma prototipi. Canlı uydu güncellemesi "
    "ve ileriye dönük otomatik tahmin henüz yoktur."
)
