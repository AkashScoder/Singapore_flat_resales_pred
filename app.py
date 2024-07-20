import streamlit as st
import joblib
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

icon = Image.open(
    "C://Users//akash//PycharmProjects//pythonProject//Singapore flat resales//hdb-resale-price-singapore-1.jpg")
st.set_page_config(
    page_title="Singapore Flat Resales | By Akash S",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': """# This dashboard app is created by Akash
                        Data has been cloned from Gov website of singapore"""})

st.sidebar.header("Dashboard")
menu_select = st.sidebar.selectbox('Select from Menu', options=['Home', 'Price Predictor'])
# Set page config
# st.set_page_config(page_title="Singapore Flat Resales", layout="wide")
if menu_select == "Home":
    # Main content for the homepage
    st.markdown("<h1 style='text-align: center; color: violet;'>Singapore Flat Resale Predictor</h1>",
                unsafe_allow_html=True)
    st.image(icon, width=400)
    # Center-align the image with adjusted width
    # st.markdown(
    #     "<div style='text-align: center;'><img src='{}' style='width: 50%; max-width: 800px;'/></div>".format(icon),
    #     unsafe_allow_html=True
    # )

    st.markdown("### :violet[Domain:] **Real Estate**")
    st.markdown("### :violet[Technologies Used:]")
    st.markdown("- NumPy")
    st.markdown("- Pandas")
    st.markdown("- Python")
    st.markdown("- Scikit-Learn")

    st.markdown("## :rocket: Overview")
    st.markdown(
        "The Flat resale price predictor is developed by analyzing data from the [Singapore government website](https://www.data.gov.sg/dataset/resale-flat-prices). Utilizing advanced data processing and machine learning techniques to make informed predictions based on historical data. This solution was designed using Streamlit for an interactive and user-friendly experience."
    )

    st.markdown("### :bulb: Key Features")
    st.markdown("- Predict resale prices of flats with high accuracy")
    st.markdown("- Analyze market trends and price fluctuations")
    # st.markdown("- Interactive visualizations for better insights")

    # Ensure everything is displayed
    st.sidebar.markdown("---")  # Separator for sidebar
    st.markdown("---")  # Separator for main content

if menu_select == 'Price Predictor':
    # Load data from joblib file
    data = joblib.load('label_mapper.pkl')

    # Separate the data into respective dictionaries
    town = data['town']
    flat_type = data['flat_type']
    block = data['block']
    street_name = data['street_name']
    storey_range = data['storey_range']
    flat_model = data['flat_model']

    # Load additional data for visualization
    dist_df = joblib.load('dist_df.pkl')

    # Create a Streamlit app
    st.markdown("""
        <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 20px; /* Add margin below title */
        }
        .button-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .button {
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            color: white;
            background-color: #4CAF50;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            text-align: center;
        }
        .output-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
            font-size: 24px; /* Increased font size */
            font-weight: bold; /* Bold text */
            text-align: center; /* Center-align text */
        }
        </style>
        <div class="title">Singapore Flat Resales</div>
        """, unsafe_allow_html=True)

    # Define dummy value
    dummy_value = "Select"

    # Create columns for inputs
    col1, col2 = st.columns(2)

    # Select month
    with col1:
        st.header('Month')
        selected_month = st.selectbox('Select the Month', [dummy_value] + [i for i in range(1, 13)],
                                      format_func=lambda x: f"{x:02d}" if x != dummy_value else dummy_value)

    # Select year
    with col2:
        st.header('Year')
        selected_year = st.selectbox('Select the Year', [dummy_value] + [i for i in range(2015, 2025)],
                                     format_func=lambda x: f"{x}" if x != dummy_value else dummy_value)

    # Select town
    with col1:
        st.header("Town")
        selected_town = st.selectbox("Select the Town name", [dummy_value] + list(town.keys()))

    # Select flat type
    with col2:
        st.header("Flat Type")
        selected_flat_type = st.selectbox("Select Flat Type", [dummy_value] + list(flat_type.keys()))

    # Select block
    with col1:
        st.header("Block")
        selected_block = st.selectbox("Select Block", [dummy_value] + list(block.keys()))

    # Select street name
    with col2:
        st.header('Street Name')
        selected_street_name = st.selectbox('Select Street Name', [dummy_value] + list(street_name.keys()))

    # Select storey range
    with col1:
        st.header('Storey Range')
        selected_storey_range = st.selectbox('Select Storey Range', [dummy_value] + list(storey_range.keys()))

    # Select flat model
    with col2:
        st.header('Flat Model')
        selected_flat_model = st.selectbox('Select Flat Model', [dummy_value] + list(flat_model.keys()))

    # Input for floor area (in sqm)
    with col1:
        st.header('Floor Area')
        floor_area_sqm = st.number_input('Enter Floor Area (sqm)', min_value=35.0, max_value=170.0, format="%.2f")

    # Input for lease commence year
    with col2:
        st.header('Lease Commence Year')
        lease_commence_year = st.number_input('Select Lease Commence Year', min_value=1966, max_value=2024, format="%d")

    # Input for remaining lease (in years)
    with col1:
        st.header('Remaining Lease')
        remaining_lease = st.number_input('Enter Remaining Lease (years)', min_value=40, max_value=100)

    # Center button in a custom container
    if st.button('Calculate Resale Price', key='calculate_resale_price'):
        try:
            scaler = joblib.load('scaler.pkl')
            xgb_model = joblib.load('xgb_model.pkl')

            # Check if selected street name exists in the dataframe
            if selected_street_name in dist_df.index:
                values = dist_df.loc[selected_street_name].values
                num_cols = [floor_area_sqm, selected_month, selected_year, lease_commence_year, remaining_lease] + list(
                    values)
                num_cols = np.array(num_cols).reshape(1, -1)  # Reshape for the scaler

                # Scale the numeric values
                scaled_values = scaler.transform(num_cols)

                # Get categorical column values and combine with scaled values
                cat_cols = np.array([
                    town[selected_town], flat_type[selected_flat_type], block[selected_block],
                    street_name[selected_street_name], storey_range[selected_storey_range],
                    flat_model[selected_flat_model]
                ]).reshape(1, -1)

                # Make prediction
                final_vector = np.hstack([cat_cols, scaled_values])
                pred = xgb_model.predict(final_vector)

                # Display calculated resale price
                st.markdown(f"""
                    <div class="output-container">
                        <p>Predicted Resale Price: SGD <span style="font-size: 28px; font-weight: bold;">{pred[0]:,.2f}</span></p>
                    </div>
                """, unsafe_allow_html=True)

            else:
                st.error(f"Street Name '{selected_street_name}' not found in data.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Add space after the button
    st.write("")
    st.write("")
