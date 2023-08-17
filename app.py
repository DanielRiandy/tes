import streamlit as st 
import pandas as pd 
import numpy as np
import pickle
import xgboost
from datetime import datetime

st.title("Customer Classifier 🔬")

st.sidebar.write(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
add_selectbox = st.sidebar.selectbox(
    'What do you want to do?',
    ('Single Prediction', 'Multiple Prediction')
)

model_ = pickle.load(open(r'Artifacts/Model/model.bin', "rb"))
enc_payor_entity_scale = pickle.load(open(r'Artifacts/Features/payor_entity_scale.b', "rb"))
enc_benef_industry = pickle.load(open(r'Artifacts/Features/benef_industry.b', "rb"))
enc_loan_status = pickle.load(open(r'Artifacts/Features/loan_status.b', "rb"))
enc_province_name = pickle.load(open(r'Artifacts/Features/province_name.b', "rb"))


if add_selectbox == 'Single Prediction':
    col1, col2 = st.columns(2)
    
    product_group = col1.selectbox(
        'Select your product Group',
        ('PO Financing', 'Invoice Financing', 'AP Financing'))
    
    benef_entity = col1.selectbox(
        'Select your Benefit Industry',
        ('Human Resource & Employment Services',
           'Trading Companies & Distributors', 'Air Freight & Logistics',
           'Distributors', 'Building Products', 'Others',
           'Wireless Telecommunication Services',
           'IT Consulting & Other Services',
           'Integrated Telecommunication Services', 'Alternative Carriers',
           'Construction & Engineering', 'Marine', 'Industrial Gases',
           'Trucking', 'Health Care Equipment', 'Personal Products',
           'Coal & Consumable Fuels', 'Health Care Distributors',
           'Research & Consulting Services', 'Diversified Support Services',
           'Oil & Gas Equipment & Services', 'Food Retail', 'Multi-Utilities',
           'Diversified Metals & Mining', 'Oil & Gas Drilling',
           'Health Care Supplies',
           'Technology Hardware, Storage & Peripherals',
           'Specialized Finance', 'Fertilizers & Agricultural Chemicals',
           'Precious Metals & Minerals', 'Auto Parts & Equipment',
           'Apparel, Accessories & Luxury Goods', 'Integrated Oil & Gas',
           'Systems Software', 'Airport Services', 'Agricultural Products',
           'Specialty Stores', 'Real Estate Development',
           'Electric Utilities', 'Internet Software & Services',
           'Heavy Electrical Equipment', 'Oil & Gas Storage & Transportation',
           'Packaged Foods & Meats', 'Airlines', 'Paper Packaging',
           'Internet & Direct Marketing Retail',
           'Electronic Manufacturing Services',
           'Specialized Consumer Services', 'Construction Materials',
           'Data Processing & Outsourced Services', 'Industrial Machinery',
           'Oil & Gas Exploration & Production', 'Home Furnishings',
           'Commodity Chemicals', 'Construction Machinery & Heavy Trucks',
           'Other Diversified Financial Services', 'Advertising',
           'Automotive Retail', 'Health Care Services', 'Leisure Products'))
    
    loan_status = col1.selectbox(
        'Select your Loan Status',
        ('Paid', 'Outstanding'))
    
    payor_entity_scale = col1.selectbox(
        'Select your Payor entity scale',
        ('National', 'BUMN', 'International', 'Multinational', 'Global',
           'BUMN Group', 'Digital Startup', 'Multinational Group',
           'Government/Ministry/Local Government'))
    
    province_name = col2.selectbox(
        'Select your province name',
        ('DKI Jakarta', 'Sumatera Selatan', 'Sulawesi Selatan',
           'Jawa Timur', 'Jawa Barat', 'Kalimantan Timur', 'Banten',
           'Jawa Tengah', 'Lampung', 'Riau', 'Sumatera Utara',
           'Kalimantan Tengah', 'Kalimantan Selatan', 'Sumatera Barat',
           'Kepulauan Bangka Belitung', 'Nusa Tenggara Barat (NTB)'))
    
    
    
    disbursement_amount = col2.number_input(
        "disbursement amount", min_value = 0, step= 1
    )
    
    ujrah_per_annum = col2.number_input(
        "ujrah per annum", min_value = 0.0, step= .1
    )
    
    tenor_in_days = col2.number_input(
        "tenor (in Days)", min_value = 0, step= 1
    )

    COL_USED_TEMP = [benef_entity, loan_status, province_name, payor_entity_scale, tenor_in_days]
    unseen_temp = np.array([COL_USED_TEMP]).reshape(1,-1)
    
    COL_USED = [enc_payor_entity_scale.transform([payor_entity_scale])[0],
                enc_benef_industry.transform([benef_entity])[0],
                enc_loan_status.transform([loan_status])[0],
                enc_province_name.transform([province_name])[0], 
                tenor_in_days]
    unseen_ = np.array([COL_USED]).reshape(1,-1)
    
    result = "None"
    if st.button(
        "Get Result"
    ):
        result = model_.predict_proba(unseen_)[0][0]
    
    if result != "None":
        if result > .5:
            pred = 'Good Customer'
            score = result
        else :
            pred = 'Bad Customer'
            score = 1 - result
        
        if 0.0 in unseen_temp:
            st.warning('Features invalid! Please fill the features with non-zero values', icon="⚠️")
        else :
            st.success(f"{pred} | Confidence Level: {round((score*100),3)}%", icon="✅")

elif add_selectbox == 'Multiple Prediction':
    uploaded_file = st.file_uploader("Choose a file (.csv, .xlsx)")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=';')
        except:
            df = pd.read_excel(uploaded_file)
            
        st.write(df)

        COL_USED = ['benef_industry', 'loan_status', 'payor_entity_scale', 'province_name', 'tenor_in_days']
        df_temp = df[COL_USED]
        
        df_temp['payor_entity_scale'] = enc_payor_entity_scale.transform(df_temp['payor_entity_scale'])
        df_temp['benef_industry'] = enc_benef_industry.transform(df_temp['benef_industry'])
        df_temp['loan_status'] = enc_loan_status.transform(df_temp['loan_status'])
        df_temp['province_name'] = enc_province_name.transform(df_temp['province_name'])
        
        df['Prediction'] = model_.predict(df_temp)
        st.write('Prediction Done ✅')

        st.download_button(
              label="Download data as CSV",
              data=df.to_csv().encode('utf-8'),
              file_name='data.csv',
              mime='text/csv',
        )
        
