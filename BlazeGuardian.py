import streamlit as st
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
