#  necessary imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# reading the dataset and accessing it using the variable df
df=pd.read_csv('updated_pollution_dataset.csv')

# Dropping records with invalid data
invalid_data_cols=['PM10','SO2']
rows_to_drop=df[(df[invalid_data_cols]<0).any(axis=1)]
df_clean=df[(df[invalid_data_cols]>=0).all(axis=1)]
df=df_clean

# Removing the outliers. Check the jupyter file for visualization
def remove_outliers(df, columns):
    outlier_counts = {}
    outlier_indices = set()  

    for i, col in enumerate(columns):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + (1.5 * IQR)
        
        column_outliers = df[df[col] > upper_bound].index
        outlier_counts[col] = len(column_outliers)
        outlier_indices.update(column_outliers)
        
    cleaned_df = df.drop(index=list(outlier_indices))
    return cleaned_df, outlier_counts, len(outlier_indices)

cols = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']
cleaned_df, outliers, total_outliers = remove_outliers(df, cols)
df=cleaned_df

# Separating features and target variables
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

# Converting categorical target variable into numeric datatype
lab_enc=LabelEncoder()
y=lab_enc.fit_transform(y)

air_quality_index={
    0: 'Good',
    1: 'Hazardous',
    2: 'Moderate',
    3: 'Poor'
}

# Splitting the training and testing data
x_test, x_train, y_test, y_train= train_test_split(
    x, y, random_state=0, test_size=0.2
)

# Creating an object of Random Forest Classifier after hyperparameter tuning. Refer Jupyter notebook for clarification
rfc=RandomForestClassifier(bootstrap= True,
                        criterion= 'gini', 
                        max_depth= 20, 
                        max_features= 'sqrt', 
                        min_samples_leaf= 1, 
                        min_samples_split= 5, 
                        n_estimators= 200)
rfc.fit(x_train, y_train)

# Function for validating the user input, whether the value lies in between min. and max. value, or whether the input is exit prompt
def validator(prompt, min_val, max_val):
    while True:
        user_input = input(prompt).strip()  

        if user_input.lower() == 'q': 
            print("Execution Terminated!")
            raise SystemExit

        try:
            val = float(user_input)  
            if min_val <= val <= max_val:
                return val
            else:
                print(f"Please enter a value inside the given range: {min_val}-{max_val}!")
        except ValueError:
            print("Invalid input! Please enter a numeric value or 'Q' to quit.")

# Function to get user input. Passes the minimum and maximum value to the validation function
def get_user_inputs():
    temp = validator("Enter the temperature (Range: 13-48 °C) or 'Q' to quit: ", 13, 48)
    humid = validator("Enter the humidity (Range: 36-113 %) or 'Q' to quit: ", 36, 113)
    pm25 = validator("Enter the PM 2.5 concentration level (Range: 0-58.1 µg/m³) or 'Q' to quit: ", 0, 58.1)
    pm10 = validator("Enter the PM 10 concentration level (Range: 0-77 µg/m³) or 'Q' to quit: ", 0, 77)
    no2 = validator("Enter the NO2 concentration level (Range: 7.4-50 ppb) or 'Q' to quit: ", 7.4, 50)
    so2 = validator("Enter the SO2 concentration level (Range: 0-27 ppb) or 'Q' to quit: ", 0, 27)
    co = validator("Enter the CO concentration level (Range: 0.65-3.05 ppm) or 'Q' to quit: ", 0.65, 3.05)
    proxy = validator("Enter the proximity of location to industrial areas (Range: 2.5-19.5 km) or 'Q' to quit: ", 2.5, 19.5)
    pop_density = validator("Enter the population density of the location (Range: 180-930 people/km²) or 'Q' to quit: ", 180, 930)

    return np.array([[temp, humid, pm25, pm10, no2, so2, co, proxy, pop_density]], dtype=object)

# Calling the input function
x_user = get_user_inputs()

# Predicting using the built model
y_user = rfc.predict(x_user)


# Printing the output
userDF = pd.DataFrame(data=x_user, columns=cols)
print("User input parameters-")
print(userDF.to_string(index=False))
print("The Air Quality for the given input parameters is:", air_quality_index[y_user[0]])