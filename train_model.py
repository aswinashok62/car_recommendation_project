import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import pickle

# Load dataset
df = pd.read_csv('car_recommendation_dataset.csv')

# Encode categorical features
label_encoders = {}
for column in ['Transmission', 'Fuel Type', 'Color']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Prepare feature matrix X and target vector y
X = df[['Number of Seats', 'Transmission', 'Fuel Type', 'Color']]
y = df['Car Name']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Nearest Neighbors model
model = NearestNeighbors(n_neighbors=1)
model.fit(X_scaled)

# Save model and preprocessing objects
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Save label encoders
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

# Save car names
with open('car_names.pkl', 'wb') as file:
    pickle.dump(y.values, file)
