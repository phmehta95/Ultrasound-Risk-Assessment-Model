from datetime import datetime
import pandas as pd
import bambi as bmb
import numpy as np
import arviz as az




#1. Data Preparation 
#Cleaning Data from imported spreadsheet

#Load the Excel spreadsheet with scanner info 
filepath = "021126_US_BH.xlsx"
df = pd.read_excel(filepath, sheet_name="BH_US")

#Timestamp to calculate todays date
today = pd.Timestamp.today().normalize() 

#List of variables from spreadsheet to include
date_cols = [
    "Last Repair Date",
    "Last PPM Date",
    "Next PPM Date",
    "Purchase Date",
    "Installation Date",
    "Battery Last Replaced",
    "Battery Last Checked",
    "Support Level"
]
#Conversion into NaT (Not A Time) so it doesn't throw an error and mess up dataframe
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

#Create the new DataFrame containing the calculated time metrics (in days)
processed_df = pd.DataFrame()

# Identification information for scanners
processed_df["Equipment No"] = df["Equipment No"]
processed_df["Serial No"] = df["Serial No"]

################################################
###########Relevant variables###################
################################################

# Time from last repair date (Today - Last Repair Date)
processed_df["Days_Since_Last_Repair"] = (
    today - df["Last Repair Date"]
).dt.days

# Time from last PPM date (Today - Last PPM Date)
processed_df["Days_Since_Last_PPM"] = (today - df["Last PPM Date"]).dt.days

# Time until next PPM date (Next PPM Date - Today) [Negative values mean overdue]
processed_df["Days_Until_Next_PPM"] = (
    df["Next PPM Date"] - today
).dt.days

# Number of days from purchase date (Today - Purchase Date)
processed_df["Days_Since_Purchase"] = (today - df["Purchase Date"]).dt.days

# Number of days from installation date (Today - Installation Date)
processed_df["Days_Since_Installation"] = (
    today - df["Installation Date"]
).dt.days

#Number of days from Battery last replaced  (Today - Battery Last Replaced)
processed_df["Days_Since_Battery_Replaced"] = (
    today - df["Battery Last Replaced"]
).dt.days

# Number of days from Battery last checked  (Today - Battery Last Checked)
processed_df["Days_Since_Battery_Checked"] = (
    today - df["Battery Last Checked"]
).dt.days


#Support level (with strings cleaned)
processed_df["Support_Level"] = df["Support Level"].astype(str).str.strip()

#Define support levels to help produce risk flag 
low_risk_support = {
    "Full In-House Support (On KPI) <DASH>",
    "Full External Contract",
}
high_risk_support = {
    "Repairs/EST Only (No PPM)",
}


def map_support_tier(val):
    if val in low_risk_support:
        return "Lower_Risk_Support"
    elif val in high_risk_support:
        return "Higher_Risk_Support"
    else:
        return np.nan

processed_df["Support_Tier"] = processed_df["Support_Level"].apply(
    map_support_tier
)

# Save processed DataFrame to new Excel file
processed_df.to_excel("model_input_metrics.xlsx", index=False)


#2. Create the risk flag - strict binary (1 = QA is a priority, 0 = Stable)
processed_df["Risk_Flag"] = (
    (processed_df["Days_Since_Installation"] > 2920 ) #Risk flag = 1 if number of days since installation is greater than 8 years
    | (processed_df["Days_Since_Last_PPM"] > 365) #Risk flag = 1 for greater than a year since last PPM
    |   (processed_df["Support_Level"] == "Repairs Only") #Risk flag = 1 for Support Level - Repairs Only
    |       (processed_df["Support_Tier"] == "Higher_Risk_Support")
).astype(int)

#Remove rows containing missing information
model_data = processed_df.dropna(
subset=["Days_Since_Installation", "Days_Since_Last_PPM", "Support_Level", "Risk_Flag"]
)
#print(processed_df["Risk_Flag"].value_counts())

#3. Define the Bayesian Model using Bambi
model = bmb.Model(
    formula = "Risk_Flag ~ Days_Since_Installation + Days_Since_Last_PPM + Support_Level",
    data = model_data,
    family = "bernoulli",
)
#Specify draws (how many data points to sample), chains (how many starting points), tune (number of trial run steps) and target_accept (length of steps to take)
risk_data = model.fit(draws=2000, tune=1000, chains=4, target_accept=0.9)

#Use model to predict posterior probabilities
model.predict(risk_data, kind="response")
pred_prob = risk_data.posterior_predictive["Risk_Flag"]

#Average across number of starting points for model and sampling points to get one probability of failure for each scanner
model_data["Predicted_Risk_Probability"] = pred_prob.mean(
    dim=["chain", "draw"]
).values

# 4. Sort probability of scanner failure from high to low
prioritized_scanners = model_data.sort_values(
    by="Predicted_Risk_Probability", ascending=False
)
# 5. Save to excel file 
prioritized_scanners.to_excel("ultrasound_qa_priorities.xlsx",index=False)