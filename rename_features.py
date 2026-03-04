# feature renaming utility for credit default dataset

# feature name mappings
FEATURE_RENAME_MAP = {
    'LIMIT_BAL': 'Credit_Limit',
    'SEX': 'Gender',
    'EDUCATION': 'Education',
    'MARRIAGE': 'Marital_Status',
    'AGE': 'Age',
    'PAY_0': 'Pay_Status_Sep',
    'PAY_2': 'Pay_Status_Aug',
    'PAY_3': 'Pay_Status_Jul',
    'PAY_4': 'Pay_Status_Jun',
    'PAY_5': 'Pay_Status_May',
    'PAY_6': 'Pay_Status_Apr',
    'BILL_AMT1': 'Bill_Sep',
    'BILL_AMT2': 'Bill_Aug',
    'BILL_AMT3': 'Bill_Jul',
    'BILL_AMT4': 'Bill_Jun',
    'BILL_AMT5': 'Bill_May',
    'BILL_AMT6': 'Bill_Apr',
    'PAY_AMT1': 'Paid_Sep',
    'PAY_AMT2': 'Paid_Aug',
    'PAY_AMT3': 'Paid_Jul',
    'PAY_AMT4': 'Paid_Jun',
    'PAY_AMT5': 'Paid_May',
    'PAY_AMT6': 'Paid_Apr',
    'default payment next month': 'Default'
}

REVERSE_RENAME_MAP = {v: k for k, v in FEATURE_RENAME_MAP.items()}

CONTINUOUS_FEATURES_NEW = [
    'Credit_Limit', 'Age',
    'Bill_Sep', 'Bill_Aug', 'Bill_Jul', 'Bill_Jun', 'Bill_May', 'Bill_Apr',
    'Paid_Sep', 'Paid_Aug', 'Paid_Jul', 'Paid_Jun', 'Paid_May', 'Paid_Apr'
]

CATEGORICAL_FEATURES_NEW = [
    'Gender', 'Education', 'Marital_Status',
    'Pay_Status_Sep', 'Pay_Status_Aug', 'Pay_Status_Jul',
    'Pay_Status_Jun', 'Pay_Status_May', 'Pay_Status_Apr'
]

DEMOGRAPHIC_FEATURES_NEW = ['Gender', 'Education']


def get_payment_status_label(value):
    # get label for payment status values
    value = int(value)
    if value == -2:
        return "No_Consumption"
    elif value == -1:
        return "Paid_Full"
    elif value == 0:
        return "Paid_Minimum"
    elif value >= 1:
        return f"{value}Mo_Late"
    else:
        return f"Status_{value}"


def get_education_label(value):
    # get label for education values
    value = int(value)
    labels = {1: "Grad_School", 2: "University", 3: "High_School", 4: "Other"}
    return labels.get(value, f"Edu_{value}")


def get_gender_label(value):
    # get label for gender values
    value = int(value)
    labels = {1: "Male", 2: "Female"}
    return labels.get(value, f"Gender_{value}")


def get_marital_label(value):
    # get label for marital status values
    value = int(value)
    labels = {1: "Married", 2: "Single", 3: "Other"}
    return labels.get(value, f"Marital_{value}")


def rename_onehot_features(encoded_names):
    # rename one-hot encoded feature names to be more intuitive
    renamed = []
    
    for name in encoded_names:
        if name.startswith('num__'):
            renamed.append(name.replace('num__', ''))
        elif name.startswith('cat__'):
            name_without_prefix = name.replace('cat__', '')
            parts = name_without_prefix.rsplit('_', 1)
            if len(parts) != 2:
                renamed.append(name_without_prefix)
                continue
            
            feature_name, value = parts
            
            try:
                if feature_name.startswith('Pay_Status_'):
                    month = feature_name.replace('Pay_Status_', '')
                    label = get_payment_status_label(value)
                    renamed.append(f"{month}_{label}")
                elif feature_name == 'Education':
                    renamed.append(get_education_label(value))
                elif feature_name == 'Gender':
                    renamed.append(get_gender_label(value))
                elif feature_name == 'Marital_Status':
                    renamed.append(get_marital_label(value))
                else:
                    renamed.append(name_without_prefix)
            except (ValueError, KeyError):
                renamed.append(name_without_prefix)
        else:
            renamed.append(name)
    
    return renamed


def rename_dataframe(df):
    # rename dataframe columns to more intuitive names
    return df.rename(columns=FEATURE_RENAME_MAP)