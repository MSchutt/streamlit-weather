import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore

df = pd.read_parquet("station_data.pq")
df_station = pd.read_csv("station_metadata.csv")

df.columns = [x.strip() for x in df.columns]

st.write("""
         # Willkommen bei ihrem persönlichen Wetter-Zuhause Finder!
         Egal, ob Sie nach Sonnenstunden, Temperatur, Wind, Niederschlag oder Höhenmetern suchen, wir helfen Ihnen dabei, den Ort mit den optimalen Bedingungen zu ermitteln. Lassen Sie uns gemeinsam Ihren persönlichen Wetterort finden!
         """)

# Fix niederschlag (-1 => no rain; but I want it in mm)
df["nied"] = df["nied"].apply(lambda x: 0 if x == -1 else x)

def clear_stationsname(name: str):
    str_name = str(name).replace("alt", "").strip()
    return str_name[0].upper() + str_name[1:].lower()

df["station_display"] = df["Stationsname"].apply(clear_stationsname)

# Aggregation per station
agg_df = df.groupby(['Stationsname']).agg({
    'sonne': 'mean',
    't': 'mean',
    'vv': 'mean',
    'nied': 'mean',
    "station_display": "first"
}).reset_index()

# Calculate z-score
z_scores = agg_df[['sonne', 't', 'vv', 'nied']].apply(zscore)

def find_place_with_weights(agg_df, weight_factors, z_scores, hoehe_filter = 0):
    df = agg_df.copy()
    z = z_scores.copy()
    station = df_station.copy()

    # Calculate combined z-scores
    df["combined_z_score"] = (z["sonne"] * (weight_factors["sonne"] / 100) +
                                  z["t"] * (weight_factors["t"] / 100) +
                                  z["vv"] * ((weight_factors["vv"] * -1) / 100) +
                                  z["nied"] * ((weight_factors["nied"] * -1) / 100)
                              )

    df = station.merge(df, left_on="Stationsname", right_on="Stationsname")
    df["Stationsname"] = df["station_display"]
    df = df.drop_duplicates(subset=["Stationsname"])
    if hoehe_filter > 0:
        df = df[df["Höhe [m]"] > hoehe_filter]
    df = df.sort_values(by=["combined_z_score"], ascending=False)
    return df

def convert_to_weight_factor(weight_factor_labels):
    mapping = {
        "sonne": { 
            "🌑 kaum Sonne": -100,
            "🌗 eher weniger Sonne": -50,
            "🤷‍♂️ egal": 0,
            "⛅️ etwas Sonne": 50,
            "😎 Sonne!": 100
        },
        "t": { "🥶 eiskalt": -100, "❄️ eher frisch": -50, "🤷‍♂️ egal": 0, "🥰 eher mild": 50, "🔥 so heiß wie möglich": 100 },
        "vv": { "🌬️ Windkanal": -100, "💨 eher viel Wind": -50, "🤷‍♂️ egal": 0, "☁️ wenig Wind": 50, "😮‍💨 kaum Wind": 100},
        "nied": { "🌧️ Platsch-Party": -100, "☔️ eher viel Regen": -50, "🤷‍♂️ egal": 0, "🌦️ wenig Regen": 50, "🌵 kaum Regen": 100 }
    }
    return {
        k: mapping[k][v] for k, v in weight_factor_labels.items()
    }


st.write("""
         ---
         
         Bitte wählen Sie jeweils aus den Optionen aus, wie wichtig Ihnen die jeweiligen Kriterien sind.
         ### Wie wichtig ist mir...
         """)

weight_factors_labels = {
    "sonne": st.radio("Sonne", ["🌑 kaum Sonne", "🌗 eher weniger Sonne", "🤷‍♂️ egal", "⛅️ etwas Sonne", "😎 Sonne!"], horizontal=True, index=4),
    "t": st.radio("Temperatur", ["🥶 eiskalt", "❄️ eher frisch", "🤷‍♂️ egal", "🥰 eher mild", "🔥 so heiß wie möglich"], horizontal=True, index=3),
    "vv": st.radio("Wind", ["🌬️ Windkanal", "💨 eher viel Wind", "🤷‍♂️ egal", "☁️ wenig Wind", "😮‍💨 kaum Wind"], horizontal=True, index=2),
    "nied": st.radio("Niederschlag", ["🌧️ Platsch-Party", "☔️ eher viel Regen", "🤷‍♂️ egal", "🌦️ wenig Regen", "🌵 kaum Regen"], horizontal=True, index=3),
}
hoehe_filter = st.slider("Minimale Höhenmeter in Meter", 0, 2500, 0, 100)

df_result = find_place_with_weights(agg_df, convert_to_weight_factor(weight_factors_labels), z_scores, hoehe_filter=hoehe_filter)

display_cols = ["station_display", "sonne", "t", "vv", "nied", "Länge [°E]", "Breite [°N]"]
rename_cols = {
    "station_display": "Ort",
    "sonne": "Ø Sonnenstunden",
    "t": "Ø Temperatur in °C",
    "vv": "Ø Windgeschwindigkeit in km/h",
    "nied": "Ø Niederschlag in mm",
    "Länge [°E]": "Längengrad",
    "Breite [°N]": "Breitengrad"
}
df_result[display_cols].rename(columns=rename_cols).head(15)

result_df = df_result[display_cols].rename(columns=rename_cols).head(15)

st.write(f"""
         ## Ihr persönliches Wetter-Zuhause ist "{result_df.iloc[0]['Ort']}"!
         Wir zeigen Ihnen hier in Echtzeit die 15 besten Orte, die Ihren Kriterien entsprechen.
         """)

map_df = df_result.rename(columns={"Breite [°N]": "lat", "Länge [°E]": "lon"}).head(15)
map_df["hex_color"] = "#ff00ff"
display_cols_df = ["station_display", "sonne", "t", "vv", "nied"]

# Round all columns to 2 decimals
df_result[display_cols_df] = df_result[display_cols_df].round(2)

st.dataframe(df_result[display_cols_df].rename(columns=rename_cols).head(15), use_container_width=False, hide_index=True)

st.write("""
         Hier ist ihr Wetter-Zuhause auf der Karte markiert:
         """)

st.map(map_df.head(1), latitude="lat",longitude="lon", zoom=6, size=5000, color="#fcba03")

st.write("""
         Mit einem Klick auf den Button können Sie sich Bilder von Ihrem Wetter-Zuhause anzeigen lassen:
         """)
st.link_button("Bilder über Google finden", "https://www.google.com/search?tbm=isch&q=" + result_df.iloc[0]["Ort"])

