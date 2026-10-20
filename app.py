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

# HARITA - DUZELTILMIS SURUM
with tab1:

    m = folium.Map(
        location=[
            filtered["enlem"].mean(),
            filtered["boylam"].mean()
        ],
        zoom_start=11,
        tiles=None
    )

    # CARTO harita altligi
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

    # NDVI renk skalasi
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

    # 1. GERCEK NDVI NOKTALARI
    for _, point in filtered.iterrows():

        value = float(point[f"NDVI_{year}"])

        popup_real = (
            f"<b>Gercek NDVI</b><br>"
            f"Yil: {year}<br>"
            f"NDVI: {value:.4f}<br>"
            f"Etki sinifi: "
            f"{int(point['etki_sinifi'])}"
        )

        folium.CircleMarker(
            location=[
                point["enlem"],
                point["boylam"]
            ],
            radius=5,
            color=scale(value),
            fill=True,
            fill_color=scale(value),
            fill_opacity=0.85,
            popup=folium.Popup(
                popup_real,
                max_width=250
            )
        ).add_to(m)

    # 2. RANDOM FOREST TAHMIN NOKTALARI
    # Sadece secenek aktifse calisir
    if show_pred and year == 2026:

        subset = preds[
            preds["sinif"].isin(selected)
        ].merge(
            points[["id", "enlem", "boylam"]],
            on="id",
            how="inner",
            validate="one_to_one"
        )

        for _, pred in subset.iterrows():

            popup_prediction = (
                "<b>Random Forest - 2026</b><br>"
                f"Gercek NDVI: "
                f"{pred['NDVI_hedef']:.4f}<br>"
                f"Tahmin NDVI: "
                f"{pred['NDVI_tahmin']:.4f}<br>"
                f"Mutlak hata: "
                f"{pred['mutlak_hata']:.4f}<br>"
                f"Etki sinifi: "
                f"{int(pred['sinif'])}"
            )

            folium.CircleMarker(
                location=[
                    pred["enlem"],
                    pred["boylam"]
                ],
                radius=9,
                color="#7c3aed",
                weight=3,
                fill=True,
                fill_color="#7c3aed",
                fill_opacity=0.30,
                tooltip="Random Forest tahmini",
                popup=folium.Popup(
                    popup_prediction,
                    max_width=280
                )
            ).add_to(m)

    # 3. NDVI RENK SKALASINI EKLE
    scale.add_to(m)

    # 4. HARITA SINIRLARINI AYARLA
    m.fit_bounds([
        [
            filtered["enlem"].min(),
            filtered["boylam"].min()
        ],
        [
            filtered["enlem"].max(),
            filtered["boylam"].max()
        ]
    ])

    # 5. STREAMLIT HARITASINI GOSTER
    st_folium(
        m,
        height=550,
        use_container_width=True,
        returned_objects=[]
    )

    st.caption(
        "Noktalar Sentinel-2 orneklerini temsil eder. "
        "Kesintisiz uydu raster haritasi degildir."
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
