import streamlit as st
import pandas as pd
import pickle

# Load the model, scaler, label encoders, and car names
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Load the dataset
df = pd.read_csv('car_recommendation_dataset.csv')

# Streamlit app
st.title('Car Recommendation System')

# Inputs
num_seats = st.selectbox('Number of Seats', [5, 7])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Electric'])
color = st.selectbox('Color', ['Red', 'Blue', 'Green', 'Black', 'White'])
budget = st.slider('Budget (in lakhs)', min_value=0, max_value=100, step=1)  # Adjust max_value as needed

# Prepare input data
input_data = pd.DataFrame({
    'Number of Seats': [num_seats],
    'Transmission': [transmission],
    'Fuel Type': [fuel_type],
    'Color': [color]
})

# Encode input data
input_data_encoded = input_data.copy()
for column in ['Transmission', 'Fuel Type', 'Color']:
    input_data_encoded[column] = label_encoders[column].transform(input_data[column])

# Scale input data
input_data_scaled = scaler.transform(input_data_encoded)

# Find matching cars
encoded_df = df.copy()
for column in ['Transmission', 'Fuel Type', 'Color']:
    encoded_df[column] = label_encoders[column].transform(df[column])
matching_cars = encoded_df[(encoded_df['Number of Seats'] == input_data_encoded['Number of Seats'][0]) &
                           (encoded_df['Transmission'] == input_data_encoded['Transmission'][0]) &
                           (encoded_df['Fuel Type'] == input_data_encoded['Fuel Type'][0]) &
                           (encoded_df['Color'] == input_data_encoded['Color'][0])
                           ]

# Filter by budget
matching_cars_within_budget = matching_cars[df.loc[matching_cars.index, 'Current Price (lakhs)'] <= budget]

# Display matching cars
if not matching_cars_within_budget.empty:
    st.write("Matching Cars within Your Budget:")
    matching_indices = matching_cars_within_budget.index
    st.write(df.loc[matching_indices, ['Car Name', 'Current Price (lakhs)']].reset_index(drop=True))
else:
    st.write("No matching cars found within your budget.")
