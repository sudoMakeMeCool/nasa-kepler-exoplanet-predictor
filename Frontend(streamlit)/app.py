import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

model = joblib.load(PROJECT_DIR / "Training" / "exoplanet_model.pkl")

st.title('NASA Kepler Exoplanet Predictor')

st.image(
    BASE_DIR / "kepler_telesccope.PNG",
    use_container_width=True
)

st.divider()

st.write('Enter the physical parameters of the target to predict its disposition')

st.header('Transit & Orbital Properties: ')

col1, col2, col3 = st.columns(3)

with col1:
    period = st.number_input('Orbital period/koi_period (days): ', value=7.834434)
    st.image(BASE_DIR / "orbital_period.png", width=200)

with col2: 
    depth = st.number_input('Transit depth/koi_depth (ppm):', value=515.0)
    st.image(BASE_DIR / "transit_depth.PNG", width=200) 

with col3: 
    prad = st.number_input('Planetary radius/koi_prad (Earth rad): ', value=2.72)
    st.image(BASE_DIR / "rad.jpg", width=200)


with st.expander('Advance Transit & Stellar Properties (Click to expand): '):
    st.write('Modify the planet and host star characterstics here.')
    col4, col5 = st.columns(2)

    # star info
    with col4:              
        impact = st.slider('Stellar crossing alignment/koi_impact: ', value=0.587)
        insol = st.number_input('Stellar flux/koi_insol (Relative to Earth): ', value=183.48)
        steff = st.number_input('Effective temp of host star/koi_steff (in K): ', value=5774.0)
        kepmag = st.number_input('Star brightness/kepmag: ', value=14.492)
        s_gravity = st.number_input('Host star surface gravity/koi_slogg (log g): ', value=4.438)
        srad = st.number_input('Stellar radius/koi_srad (Relative to Sun): ', value=1.0)

    # planet and transit info
    with col5: 
        duration = st.number_input('Transit duration/koi_duration (in hours): ', value=3.89)
        teq = st.number_input("Planet's equillibrium temp/koi_teq (in K): ", value=939.0)
        tce_plntn = st.number_input('Threshold crossing event planet number/koi_tce_plnt_num: ', value=1)
        transit_snr = st.number_input('Transit model signal-to-noise ratio/koi_model_snr: ', value=32.1)

threshold = st.slider('Detection Strictness (Lower % catches more planets, but increases False Positives): ', min_value=0, max_value=100, value=35) / 100

if st.button('Predict Disposition'):
    input_data = pd.DataFrame([{
        'koi_period': period, 
        'koi_impact': impact, 
        'koi_duration': duration, 
        'koi_depth': depth, 
        'koi_prad': prad, 
        'koi_teq': teq,
        'koi_insol': insol, 
        'koi_model_snr': transit_snr, 
        'koi_tce_plnt_num': tce_plntn, 
        'koi_steff': steff, 
        'koi_slogg': s_gravity, 
        'koi_srad': srad, 
        'koi_kepmag': kepmag
    }])

    # prob predictions
    prediction = model.predict_proba(input_data)

    # converting them to absolute 0 and 1 and final output
    if (prediction[0, 1] >= threshold):
        st.success(f'CONFIRMED EXOPLANET! 🪐 \n Model Confidence: {prediction[0, 1] * 100: .2f}%')
    else:
        st.error(f'FALSE POSITIVE. ❌ \n Model Confidence: {prediction[0, 0] * 100: .2f}')