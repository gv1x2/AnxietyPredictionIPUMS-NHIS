import pandas as pd

file_path = 'Book11.csv'
df = pd.read_csv(file_path, delimiter=';')

# Perform Min-Max normalization on the 'AGE' column and replace the old 'AGE' column with the normalized one
df['AGE'] = (df['AGE'] - 18) / (85 - 18)
df['AGE'] = df['AGE'].round(2)

# Adjust SEX variable from 1,2 to 0,1
df['SEX'] = df['SEX'] - 1

# Group sexual orientation into 2 groups: Straight (0) and LGBT (1) and replace the old 'SEXORIEN' variable
df['SEXORIEN'] = df['SEXORIEN'].apply(lambda x: 0 if x == 2 else 1)

# Define a mapping function for the MARSTLEG categories
def map_marstleg_category(x):
    if x in [11, 12, 13]:  # Married
        return 'Married'
    elif x in [20, 30, 40]:  # Previously Married
        return 'PreviouslyMarried'
    elif x == 50:  # Never Married
        return 'NeverMarried'
    else:  # Fallback for any undefined categories
        return 'Other'

# Apply the mapping function
df['MARSTLEG_CATEGORY'] = df['MARSTLEG'].apply(map_marstleg_category)

# Apply one-hot encoding to the new 'MARSTLEG_CATEGORY' column
df = pd.concat([df, pd.get_dummies(df['MARSTLEG_CATEGORY'], prefix='Marstleg')], axis=1)

# Define a mapping function for the EDUC categories into broader education groups
def map_education_category(code):
    if code <= 116:
        return 'LessThanHighSchool'
    elif 200 <= code <= 202:
        return 'HighSchoolOrEquivalent'
    elif 300 <= code <= 303:
        return 'SomeCollegeNoDegree'
    elif code == 400:
        return 'BachelorsDegree'
    elif 500 <= code:
        return 'GraduateDegree'

# Apply the mapping function to create a new 'EDUC_GROUP' column
df['EDUC_GROUP'] = df['EDUC'].apply(map_education_category)

# Apply one-hot encoding to the new 'EDUC_GROUP' column
df = pd.concat([df, pd.get_dummies(df['EDUC_GROUP'], prefix='Edu')], axis=1)

# Drop the original 'MARSTLEG', 'EDUC', 'MARSTLEG_CATEGORY', and 'EDUC_GROUP' columns as they're no longer needed
df.drop(['MARSTLEG', 'EDUC', 'MARSTLEG_CATEGORY', 'EDUC_GROUP'], axis=1, inplace=True)

def assign_value(category):
    mapping = {
        11: 0,   # Under 0.50
        12: 0.5, # 0.50 to 0.74
        13: 0.75,# 0.75 to 0.99
        21: 1,   # 1.00 to 1.24
        22: 1.25,# 1.25 to 1.49
        23: 1.5, # 1.50 to 1.74
        24: 1.75,# 1.75 to 1.99
        31: 2,   # 2.00 to 2.49
        32: 2.5, # 2.50 to 2.99
        33: 3,   # 3.00 to 3.49
        34: 3.5, # 3.50 to 3.99
        35: 4,   # 4.00 to 4.49
        36: 4.5, # 4.50 to 4.99
        37: 5    # 5.00 and over
    }
    return mapping.get(category, 0)

# Apply the function to assign values
df['RangeVar_Value'] = df['POVERTY'].apply(assign_value)

# Normalize these values to a 0 to 1 scale
df['RangeVar_Normalized'] = df['RangeVar_Value'] / 5

# Drop the temporary 'RangeVar_Value' as it's no longer needed
df.drop(['RangeVar_Value', 'POVERTY'], axis=1, inplace=True)

# Define a mapping function for OWNERSHIP
def map_ownership_category(code):
    if code == 10:  # Owned or being bought
        return 1  # Owned
    elif code == 20 or code == 30:  # Rented or Other arrangement
        return 0  # Rented

# Apply the mapping function
df['OWNERSHIP'] = df['OWNERSHIP'].apply(map_ownership_category)

# Assign numeric values based on the category, starting from 0 for 'Less than 1 year' to 20 for 'More than 20 years'
def assign_years_lived_value(category):
    if category == 1:   # 'Less than 1 year'
        return 0
    elif category == 2: # '1 to 3 years'
        return 1
    elif category == 3: # '4 to 10 years'
        return 4
    elif category == 4: # '11 to 20 years'
        return 11
    elif category == 5: # 'More than 20 years'
        return 20
    else:
        return None

# Apply function to assign values
df['HOUYRSLIV_Value'] = df['HOUYRSLIV'].apply(assign_years_lived_value)

# Normalize the values to a 0-1 scale
df['HOUYRSLIV_Normalized'] = df['HOUYRSLIV_Value'] / 20

# Drop the temporary 'HOUYRSLIV_Value' column
df.drop(['HOUYRSLIV_Value','HOUYRSLIV'], axis=1, inplace=True)

# Ordinal encoding for 'HEALTH' is already in a suitable format (1-5), 
# but you might want to normalize it to a 0-1 range for consistency with other variables
df['HEALTH'] = (df['HEALTH'] - 1) / (5 - 1)

# Normalize 'HEIGHT' from 59 to 76 inches to a 0-1 scale
df['HEIGHT'] = (df['HEIGHT'] - 59) / (76 - 59)
df['HEIGHT'] = df['HEIGHT'].round(2)
# Normalize 'WEIGHT' from 100 to 299 pounds to a 0-1 scale
df['WEIGHT'] = (df['WEIGHT'] - 100) / (299 - 100)
df['WEIGHT'] = df['WEIGHT'].round(2)

# This creates a new dataframe with one-hot encoded columns for each category
smoking_dummies = pd.get_dummies(df['SMOKESTATUS2'], prefix='Smoking')

# Concatenate the original dataframe with the new one-hot encoded columns
df = pd.concat([df, smoking_dummies], axis=1)

# Optionally, drop the original 'SMOKING_STATUS' column if you no longer need it
df.drop('SMOKESTATUS2', axis=1, inplace=True)

df['ANXIETYEV'] = df['ANXIETYEV'] - 1

# Apply one-hot encoding to 'WORFREQ'
worfreq_dummies = pd.get_dummies(df['WORFREQ'], prefix='WorryFreq')

# Concatenate the original DataFrame with the new one-hot encoded columns
df = pd.concat([df, worfreq_dummies], axis=1)
df.drop('WORFREQ', axis=1, inplace=True)

# Apply one-hot encoding to 'DEPFREQ'
depfreq_dummies = pd.get_dummies(df['DEPFREQ'], prefix='DepFreq')

# Concatenate the original DataFrame with the new one-hot encoded columns
df = pd.concat([df, depfreq_dummies], axis=1)
df.drop('DEPFREQ', axis=1, inplace=True)


# Save the DataFrame back to a new CSV file with ';' as the delimiter
df.to_csv('Book27.csv', sep=';', index=False)

# Display a message to confirm file has been saved
print("File has been saved with the updated 'AGE', 'SEXORIEN', 'MARSTLEG', and 'EDUC' variables as separate binary indicators.")

#we need to identify/predict anxiety. what type of neural networks or other algorithms we can try?