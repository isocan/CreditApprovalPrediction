import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import altair as alt

load_xgb = pickle.load(open('xgb_model.pkl', 'rb'))

# Load your data and process it as needed
# For example, load the merged DataFrame containing credit scores and features
df = pd.read_csv('total_client_data_final.csv')

index_col = 'SK_ID_CURR'

df = df.set_index(index_col)
features = df.columns[:-1]

# Determine the minimum and maximum client IDs
min_client_id = df.index.min()
max_client_id = df.index.max()

st.markdown(
    f"<h1 style='color:#001f3f'>Prévision d'Approbation de Crédit et Analyse SHAP du Profil</h1>",
    unsafe_allow_html=True
)


# Initialize client_id and selected_score to default values
client_id = min_client_id
selected_score = 0.0

# Add a text input for the user to enter a client ID
client_id = st.number_input(f'Entrez un numéro de client entre {min_client_id} et {max_client_id}', min_client_id, max_client_id, client_id)

if client_id:
    # Validate that the entered client ID is within the valid range
    client_id = int(client_id)
    if min_client_id <= client_id <= max_client_id:
        # Calculate the credit score and related information here
        selected_score = df[df.index == client_id]['credit_score'].iloc[0]

        # Calculate the normalized distance
        threshold = 50
        distance = selected_score - threshold
        distance_normalisée = max(0.0, min(1.0, (distance + 100) / 200))

        # Define the color based on whether the score is accepted or not
        if distance >= 0:
            couleur = 'green'
            étiquette = 'Accepté'
            symbole = '✔️'
        else:
            couleur = 'red'
            étiquette = 'Non Accepté'
            symbole = '❌'

        # Display the credit score with the color, symbol, and label
        st.write(f"<p style='color:{couleur}'>Score de Crédit : <b>{selected_score:.2f}</b></p>", unsafe_allow_html=True)
        st.write(f"<p style='color:{couleur}'>{symbole} {étiquette}</p>", unsafe_allow_html=True)

        st.progress(distance_normalisée)

    else:
        st.write("Veuillez entrer un numéro de client valide dans la plage spécifiée.")

explainer = shap.TreeExplainer(load_xgb)

# Get the data for the selected index
selected_data = df[df.index == client_id]
selected_data = selected_data.iloc[:, :-1]

# Calculate the credit score and feature importance
X_sample = selected_data.values.reshape(1, -1)
shap_values = explainer.shap_values(X_sample)

# Create a DataFrame with SHAP values and corresponding column names
shap_df = pd.DataFrame(data=shap_values, columns=features)

# Get the top 10 absolute SHAP values with their corresponding column names
top_10_shap = shap_df.abs().stack().sort_values(ascending=False).head(10)

# # Display the top 10 absolute SHAP values with column names
# st.write('Top 10 Absolute SHAP Values with Column Names:')
# st.write(top_10_shap)

# Plot local feature importance sorted
abs_shap_values = np.abs(shap_values)
sorted_indices = np.argsort(abs_shap_values)
N = 10
top_feature_indices = sorted_indices[0, -N:]
top_feature_names = [features[i] for i in top_feature_indices]
top_feature_shap_values = shap_values[0, top_feature_indices]

# Create a DataFrame for the top features and their SHAP values
top_features_df = pd.DataFrame({'Feature': top_feature_names, 'SHAP Value': top_feature_shap_values})

st.write("Top {} Most Important Features:".format(N))

# Create an Altair chart
chart = alt.Chart(top_features_df).mark_bar().encode(
    x='SHAP Value',
    y=alt.Y('Feature', sort='-x'),
    color=alt.condition(
        alt.datum['SHAP Value'] > 0,
        alt.value('green'),  # Color for positive SHAP values
        alt.value('red')     # Color for negative SHAP values
    )
).properties(width=600, height=400)

st.altair_chart(chart)

# Create a dropdown to select a feature to plot
selected_feature = st.selectbox("Select a feature to plot", top_feature_names)

# Percentage of data to use (e.g., 10%)
data_percentage = 0.1

# Sample the non-selected client data
non_selected_data = df[df.index != client_id]
sampled_data = non_selected_data.sample(frac=data_percentage, random_state=42)

# Include the data of the selected client
selected_client_data = df[df.index == client_id]
sampled_data = pd.concat([selected_client_data, sampled_data])

# Create a histogram for the selected feature
histogram = alt.Chart(sampled_data).mark_bar().encode(
    x=alt.X(selected_feature, bin=True),
    y='count()',
    color=alt.condition(
        alt.datum.index == client_id,
        alt.value('red'),  # Color for the selected client
        alt.value('blue')  # Color for other clients
    )
).properties(width=600, height=400)

# Add a vertical line to indicate the position of the selected client's data
vertical_line = alt.Chart(selected_client_data).mark_rule(color='red').encode(
    x=selected_feature
)

st.altair_chart(histogram + vertical_line)





