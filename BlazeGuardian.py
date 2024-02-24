import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import confusion_matrix

st.title("_BlazeGuardian_")

st.write("""
## Explore the factors that contribute to the wildfires in Alberta and Beyond
This machine learning model not only displays to you how  much each factor
contributes to a wildfire, but also allows you to explore the data further by
segmenting it into various causes and possible factors to help mitigate the risk
of wildfires particularly in first nations communities. Please download the
dataset below for your reference.
""")

st.download_button("Alberta Wildfires Data Set", "/content/fp-historical-wildfire-data-2006-2021.csv", file_name = "fp-historical-wildfire-data-2006-2021.csv")

st.subheader("Access to Detailed Report")
st.write("Use the button below to acces a in-depth analysis of our findings.")
st.link_button("Access Report", "https://docs.google.com/document/d/1frUK1DXamQg05Hj76eCshZuxbO8vuOiJT_kCOUjOk_4/edit?usp=sharing")

st.sidebar.title("_Explore the dataset!_")


st.write("""
The input variables provided to the prediction function represent key characteristics of a fire incident, including its geographical location (latitude and longitude), potential causes such as recreation or lightning strikes, fire spread rate, fuel types involved, and activity classes related to the fire. The XGBoost model leverages these features to predict a severity index value, indicating the anticipated severity or extent of the fire incident. This index value serves as a quantitative or qualitative measure of the fire's severity, aiding in decision-making and resource allocation for firefighting and emergency response efforts.
""")

url = "https://raw.githubusercontent.com/surajs45/BlazeGuardian/main/Wildfire_Preprocessed_Data.csv"
df1 = pd.read_csv(url, index_col=0)

st.dataframe(df1)

loaded_model = xgb.XGBClassifier()
loaded_model.load_model("https://raw.githubusercontent.com/surajs45/BlazeGuardian/main/model_with_coordinates.json")

required_columns = ['fire_location_latitude', 'fire_location_longitude', 'general_cause_desc_Recreation',
                    'fire_spread_rate', 'fuel_type_C2', 'fuel_type_O1a', 'activity_class_Cooking and Warming',
                    'general_cause_desc_Lightning']


X_example = df1[required_columns].copy()

example_pred = loaded_model.predict(X_example)


# Define the prediction function
def predict_fire_size(latitude, longitude, general_cause_desc_Recreation, fire_spread_rate,
                      fuel_type_C2, fuel_type_O1a, activity_class_Cooking, general_cause_desc_Lightning):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'fire_location_latitude': [latitude],
        'fire_location_longitude': [longitude],
        'general_cause_desc_Recreation': [general_cause_desc_Recreation],
        'fire_spread_rate': [fire_spread_rate],
        'fuel_type_C2': [fuel_type_C2],
        'fuel_type_O1a': [fuel_type_O1a],
        'activity_class_Cooking and Warming': [activity_class_Cooking],
        'general_cause_desc_Lightning': [general_cause_desc_Lightning]
    })

    # Make prediction using the loaded model
    prediction = loaded_model.predict(input_data)

    # Return the prediction
    return prediction[0]

# Streamlit UI
st.title('Fire Size Prediction')

# Input fields for prediction
latitude = st.number_input('Latitude')
longitude = st.number_input('Longitude')
general_cause_desc_Recreation = st.selectbox('General Cause Recreation', [0, 1])
fire_spread_rate = st.number_input('Fire Spread Rate')
fuel_type_C2 = st.selectbox('Fuel Type C2', [0, 1])
fuel_type_O1a = st.selectbox('Fuel Type O1a', [0, 1])
activity_class_Cooking = st.selectbox('Activity Class Cooking and Warming', [0, 1])
general_cause_desc_Lightning = st.selectbox('General Cause Lightning', [0, 1])

# Button to trigger prediction
if st.button('Predict Fire Size'):
    # Call the prediction function
    prediction = predict_fire_size(latitude, longitude, general_cause_desc_Recreation, fire_spread_rate,
                                   fuel_type_C2, fuel_type_O1a, activity_class_Cooking, general_cause_desc_Lightning)

    # Display the prediction
    st.subheader(f'Predicted Fire Size: {prediction}')

    longitudes = range(0, 120)
    latitudes = range(0, 120)
    predictions = [predict_fire_size(latitude, longitude, general_cause_desc_Recreation, fire_spread_rate,
                                     fuel_type_C2, fuel_type_O1a, activity_class_Cooking,
                                     general_cause_desc_Lightning) for latitude in latitudes]

    # Plot the bar graph
    fig, ax = plt.subplots()
    ax.bar(latitudes, predictions)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Predicted Fire Size')
    ax.set_title('Predicted Fire Size vs Latitude')
    st.pyplot(fig)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("BlazeGuardian")

st.write("""
## Explore the factors that contribute to wildfires in Alberta and Beyond
This machine learning model not only displays how much each factor contributes to a wildfire, but also allows you to explore the data further by segmenting it into various causes and possible factors to help mitigate the risk of wildfires, particularly in First Nations communities. Please view to the dataset below for your reference.
""")

df1 = pd.read_csv("https://raw.githubusercontent.com/surajs45/BlazeGuardian/main/Wildfire_Preprocessed_Data.csv")
y_fields = pd.read_csv("Wildfire_Preprocessed_Actual.csv")

st.dataframe(df1)

X = df1
y = y_fields['size_class']

data1 = pd.DataFrame(df1["fire_location_longitude"], df1["fire_location_latitude"])

lon_lat_table = df1.rename(columns={'fire_location_latitude': 'LAT', 'fire_location_longitude': 'LON'})

st.dataframe(lon_lat_table)

st.subheader("The map below displays hot spots for fires within Alberta")
st.write("It can be seen that fires occur in all surrounding areas from Edmonton except for the South Eastern Region of the Province.")


st.map(lon_lat_table)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model_size_class = xgb.XGBClassifier(
    objective='multi:softprob',
    n_estimators=50,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

model_size_class.fit(X_train, y_train)

y_pred = model_size_class.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

class_labels = label_encoder.inverse_transform(range(len(cm)))

def calculate_metrics(cm):
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    pod = tp / (tp + fn) if (tp + fn) else 0
    pofd = fp / (fp + tn) if (fp + tn) else 0
    pss = pod - pofd
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    return pod, pofd, pss, accuracy

overall_accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) else 0

st.subheader(f"The model displays the size class through XGB classification - Overall Accuracy: {overall_accuracy:.2f}")

def find_important_features(boosting_model, X_train, limit):
    features_all = boosting_model.feature_importances_
    dataset_all_columns = X_train.columns.values[features_all > limit].reshape(-1, 1)
    features_all_trimmed = features_all[features_all > limit].reshape(-1, 1)
    indices = np.arange(0, len(features_all), 1)[features_all > limit].reshape(-1, 1)
    ranked_features = np.concatenate([indices, dataset_all_columns, features_all_trimmed], axis=1)
    sorted_ranked_features = sorted(ranked_features.tolist(), key=lambda kvvp: kvvp[2])[::-1]
    return sorted_ranked_features

model_size_class_important_features = find_important_features(
    boosting_model=model_size_class,
    X_train=X_train,
    limit=0.01
)

fig1 = plt.figure(figsize=(9, 7))
plt.bar(range(len(model_size_class.feature_importances_)), model_size_class.feature_importances_)
st.pyplot(fig1)


X = df1[[feature[1] for feature in model_size_class_important_features]]
y = y_fields['size_class']

# Encode categorical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model_size_class_important_features_only = xgb.XGBClassifier(
    objective='multi:softprob',
    n_estimators=50,            # Number of boosting rounds (you can adjust this)
    learning_rate=0.1,           # Learning rate (you can adjust this)
    max_depth=4,                 # Maximum depth of each tree (you can adjust this)
    random_state=42
)

model_size_class_important_features_only.fit(X_train, y_train)

y_pred = model_size_class_important_features_only.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Get the original class labels from LabelEncoder
class_labels = label_encoder.inverse_transform(range(len(cm)))

# Function to calculate metrics
def calculate_metrics(cm):
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    pod = tp / (tp + fn) if (tp + fn) else 0  # Probability of Detection
    pofd = fp / (fp + tn) if (fp + tn) else 0  # Probability of False Detection
    pss = pod - pofd  # Peirce Skill Score
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0  # Accuracy
    return pod, pofd, pss, accuracy

# Plot confusion matrices and metrics for each category
for i, label in enumerate(class_labels):
    # Construct a binary confusion matrix for the current category
    tp = cm[i, i]
    fn = np.sum(cm[i, :]) - tp
    fp = np.sum(cm[:, i]) - tp
    tn = np.sum(cm) - (fp + fn + tp)
    binary_cm = np.array([[tn, fp], [fn, tp]])

    # Calculate metrics
    pod, pofd, pss, accuracy = calculate_metrics(binary_cm)


# Calculate overall accuracy from the combined confusion matrix
overall_accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) else 0

# Display overall accuracy
st.subheader(f"The model displays the size class based on important features through XGB classification - Overall Accuracy: {overall_accuracy:.2f}")


model_size_class_important_features_only_important_features_high_limit = find_important_features(
    boosting_model=model_size_class_important_features_only,
    X_train=X_train,
    limit=0.03
)

plt.bar(range(len(model_size_class_important_features_only.feature_importances_)), model_size_class_important_features_only.feature_importances_)
st.pyplot()

X = df1[[feature[1] for feature in model_size_class_important_features_only_important_features_high_limit]]
y = y_fields['size_class']

# Encode categorical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model_size_class_most_important_features_only = xgb.XGBClassifier(
    objective='multi:softprob',
    n_estimators=50,            # Number of boosting rounds (you can adjust this)
    learning_rate=0.1,           # Learning rate (you can adjust this)
    max_depth=4,                 # Maximum depth of each tree (you can adjust this)
    random_state=42
)

model_size_class_most_important_features_only.fit(X_train, y_train)

y_pred = model_size_class_most_important_features_only.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Get the original class labels from LabelEncoder
class_labels = label_encoder.inverse_transform(range(len(cm)))

# Function to calculate metrics
def calculate_metrics(cm):
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    pod = tp / (tp + fn) if (tp + fn) else 0  # Probability of Detection
    pofd = fp / (fp + tn) if (fp + tn) else 0  # Probability of False Detection
    pss = pod - pofd  # Peirce Skill Score
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0  # Accuracy
    return pod, pofd, pss, accuracy

# Plot confusion matrices and metrics for each category
for i, label in enumerate(class_labels):
    # Construct a binary confusion matrix for the current category
    tp = cm[i, i]
    fn = np.sum(cm[i, :]) - tp
    fp = np.sum(cm[:, i]) - tp
    tn = np.sum(cm) - (fp + fn + tp)
    binary_cm = np.array([[tn, fp], [fn, tp]])

    # Calculate metrics
    pod, pofd, pss, accuracy = calculate_metrics(binary_cm)



# Calculate overall accuracy from the combined confusion matrix
overall_accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) else 0

# Display overall accuracy
st.subheader(f"The model displays the size class based on most important features only through XGB classification - Overall Accuracy: {overall_accuracy:.2f}")

model_size_class_most_important_features_only_important_features = find_important_features(
    boosting_model=model_size_class_most_important_features_only,
    X_train=X_train,
    limit=0.01
)

fig2 = plt.figure(figsize=(9, 7))
plt.bar(range(len(model_size_class_most_important_features_only.feature_importances_)), model_size_class_most_important_features_only.feature_importances_)
st.pyplot(fig2)
