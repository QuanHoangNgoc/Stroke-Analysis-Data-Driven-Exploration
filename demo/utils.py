target_col = "stroke"

def format_form(df):
    if 'id' in df.columns:
        df = df.drop(columns=["id"])
    if 'gender' in df.columns:
        df = df.drop(df[df["gender"] == "Other"].index)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df.columns = df.columns.str.lower()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].str.lower()
    for column in df.select_dtypes(include=['number']).columns:
        if(df[column].nunique() == 2):
            df[column] = df[column].astype(bool)
            df[column] = df[column].replace({True: 'yes', False: 'no'})

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    new_column_order = categorical_cols + numerical_cols
    new_column_order.remove(target_col)
    new_column_order.append(target_col)
    df = df[new_column_order]

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

def get_pass_data(X_train, y_train, is_train=True):
    X, y = X_train.copy(), y_train.copy()

    y = y.replace({"yes": 1, "no": 0})

    X = X.replace({"yes": 1, "no": 0})
    if ('gender' in X.columns):
        X['gender'] = X['gender'].map({
            'male': 0,
            'female': 1,
        }).astype('int')
    if('residence_type' in X.columns):
        X['residence_type'] = X['residence_type'].map({
            'urban': 0,
            'rural': 1,
        }).astype('int')
    if ('work_type' in X.columns):
        X['work_type'] = X['work_type'].map({
            'private': 0,
            'self-employed': 1,
            'govt_job': 2,
            'children': 3,
            'never_worked': 4,
        }).astype('int')    
    if ('smoking_status' in X.columns):
        X['smoking_status'] = X['smoking_status'].map({
            'never smoked': 0,
            'formerly smoked': 1,
            'smokes': 2,
            'unknown': 3,
        }).astype('int')
                
    return X, y
