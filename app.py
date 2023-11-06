import streamlit as st
import pandas as pd
import numpy as np


# Code from scipy, because streamlit cloud does not install it by default
def _isconst(x):
    """
    Check if all values in x are the same.  nans are ignored.

    x must be a 1d array.

    The return value is a 1d array with length 1, so it can be used
    in np.apply_along_axis.
    """
    y = x[~np.isnan(x)]
    if y.size == 0:
        return np.array([True])
    else:
        return (y[0] == y).all(keepdims=True)

def _first(arr, axis):
    """Return arr[..., 0:1, ...] where 0:1 is in the `axis` position."""
    return np.take_along_axis(arr, np.array(0, ndmin=arr.ndim), axis)

def _contains_nan(a, nan_policy='propagate', use_summation=True):
    if not isinstance(a, np.ndarray):
        use_summation = False  # some array_likes ignore nans (e.g. pandas)
    policies = ['propagate', 'raise', 'omit']
    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))

    if np.issubdtype(a.dtype, np.inexact):
        # The summation method avoids creating a (potentially huge) array.
        if use_summation:
            with np.errstate(invalid='ignore', over='ignore'):
                contains_nan = np.isnan(np.sum(a))
        else:
            contains_nan = np.isnan(a).any()
    elif np.issubdtype(a.dtype, object):
        contains_nan = False
        for el in a.ravel():
            # isnan doesn't work on non-numeric elements
            if np.issubdtype(type(el), np.number) and np.isnan(el):
                contains_nan = True
                break
    else:
        # Only `object` and `inexact` arrays can have NaNs
        contains_nan = False

    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    return contains_nan, nan_policy

def _quiet_nanstd(x, ddof=0):
    """
    Compute nanstd for the 1d array x, but quietly return nan if x is all nan.

    The return value is a 1d array with length 1, so it can be used
    in np.apply_along_axis.
    """
    y = x[~np.isnan(x)]
    if y.size == 0:
        return np.array([np.nan])
    else:
        return np.std(y, keepdims=True, ddof=ddof)

def _quiet_nanmean(x):
    """
    Compute nanmean for the 1d array x, but quietly return nan if x is all nan.

    The return value is a 1d array with length 1, so it can be used
    in np.apply_along_axis.
    """
    y = x[~np.isnan(x)]
    if y.size == 0:
        return np.array([np.nan])
    else:
        return np.mean(y, keepdims=True)

def zmap(scores, compare, axis=0, ddof=0, nan_policy='propagate'):
    """
    Calculate the relative z-scores.

    Return an array of z-scores, i.e., scores that are standardized to
    zero mean and unit variance, where mean and variance are calculated
    from the comparison array.

    Parameters
    ----------
    scores : array_like
        The input for which z-scores are calculated.
    compare : array_like
        The input from which the mean and standard deviation of the
        normalization are taken; assumed to have the same dimension as
        `scores`.
    axis : int or None, optional
        Axis over which mean and variance of `compare` are calculated.
        Default is 0. If None, compute over the whole array `scores`.
    ddof : int, optional
        Degrees of freedom correction in the calculation of the
        standard deviation. Default is 0.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle the occurrence of nans in `compare`.
        'propagate' returns nan, 'raise' raises an exception, 'omit'
        performs the calculations ignoring nan values. Default is
        'propagate'. Note that when the value is 'omit', nans in `scores`
        also propagate to the output, but they do not affect the z-scores
        computed for the non-nan values.

    Returns
    -------
    zscore : array_like
        Z-scores, in the same shape as `scores`.

    Notes
    -----
    This function preserves ndarray subclasses, and works also with
    matrices and masked arrays (it uses `asanyarray` instead of
    `asarray` for parameters).

    Examples
    --------
    >>> from scipy.stats import zmap
    >>> a = [0.5, 2.0, 2.5, 3]
    >>> b = [0, 1, 2, 3, 4]
    >>> zmap(a, b)
    array([-1.06066017,  0.        ,  0.35355339,  0.70710678])

    """
    a = np.asanyarray(compare)

    if a.size == 0:
        return np.empty(a.shape)

    contains_nan, nan_policy = _contains_nan(a, nan_policy)

    if contains_nan and nan_policy == 'omit':
        if axis is None:
            mn = _quiet_nanmean(a.ravel())
            std = _quiet_nanstd(a.ravel(), ddof=ddof)
            isconst = _isconst(a.ravel())
        else:
            mn = np.apply_along_axis(_quiet_nanmean, axis, a)
            std = np.apply_along_axis(_quiet_nanstd, axis, a, ddof=ddof)
            isconst = np.apply_along_axis(_isconst, axis, a)
    else:
        mn = a.mean(axis=axis, keepdims=True)
        std = a.std(axis=axis, ddof=ddof, keepdims=True)
        if axis is None:
            isconst = (a.item(0) == a).all()
        else:
            isconst = (_first(a, axis) == a).all(axis=axis, keepdims=True)

    # Set std deviations that are 0 to 1 to avoid division by 0.
    std[isconst] = 1.0
    z = (scores - mn) / std
    # Set the outputs associated with a constant input to nan.
    z[np.broadcast_to(isconst, z.shape)] = np.nan
    return z


def zscore(a, axis=0, ddof=0, nan_policy='propagate'):
    """
    Compute the z score.

    Compute the z score of each value in the sample, relative to the
    sample mean and standard deviation.

    Parameters
    ----------
    a : array_like
        An array like object containing the sample data.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.
    ddof : int, optional
        Degrees of freedom correction in the calculation of the
        standard deviation. Default is 0.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.  Note that when the value is 'omit',
        nans in the input also propagate to the output, but they do not affect
        the z-scores computed for the non-nan values.

    Returns
    -------
    zscore : array_like
        The z-scores, standardized by mean and standard deviation of
        input array `a`.

    Notes
    -----
    This function preserves ndarray subclasses, and works also with
    matrices and masked arrays (it uses `asanyarray` instead of
    `asarray` for parameters).

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([ 0.7972,  0.0767,  0.4383,  0.7866,  0.8091,
    ...                0.1954,  0.6307,  0.6599,  0.1065,  0.0508])
    >>> from scipy import stats
    >>> stats.zscore(a)
    array([ 1.1273, -1.247 , -0.0552,  1.0923,  1.1664, -0.8559,  0.5786,
            0.6748, -1.1488, -1.3324])

    Computing along a specified axis, using n-1 degrees of freedom
    (``ddof=1``) to calculate the standard deviation:

    >>> b = np.array([[ 0.3148,  0.0478,  0.6243,  0.4608],
    ...               [ 0.7149,  0.0775,  0.6072,  0.9656],
    ...               [ 0.6341,  0.1403,  0.9759,  0.4064],
    ...               [ 0.5918,  0.6948,  0.904 ,  0.3721],
    ...               [ 0.0921,  0.2481,  0.1188,  0.1366]])
    >>> stats.zscore(b, axis=1, ddof=1)
    array([[-0.19264823, -1.28415119,  1.07259584,  0.40420358],
           [ 0.33048416, -1.37380874,  0.04251374,  1.00081084],
           [ 0.26796377, -1.12598418,  1.23283094, -0.37481053],
           [-0.22095197,  0.24468594,  1.19042819, -1.21416216],
           [-0.82780366,  1.4457416 , -0.43867764, -0.1792603 ]])

    An example with `nan_policy='omit'`:

    >>> x = np.array([[25.11, 30.10, np.nan, 32.02, 43.15],
    ...               [14.95, 16.06, 121.25, 94.35, 29.81]])
    >>> stats.zscore(x, axis=1, nan_policy='omit')
    array([[-1.13490897, -0.37830299,         nan, -0.08718406,  1.60039602],
           [-0.91611681, -0.89090508,  1.4983032 ,  0.88731639, -0.5785977 ]])
    """
    return zmap(a, a, axis=axis, ddof=ddof, nan_policy=nan_policy)
# <END Code from scipy, because streamlit cloud does not install it by default >


# Begin Analysis

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


st.write("""
         ---
         
         Erstellt von Markus Schüttengruber für den Storytelling Kurs an der FH St. Pölten.
         Ihr persönliches Wetter-Zuhause wird mittels historischen Daten ermittelt. Die Daten stammen von [GeoSphere](https://www.geosphere.at/) und sind im Zeitraum zwischen Jänner 2000 und Dezember 2022 erhoben worden.
         Vielen Dank an GeoSphere für die Bereitstellung der qualitative hochwertigen Daten!
         """)