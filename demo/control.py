import os
import pickle
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import google.generativeai as genai
from utils import *

api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)
cols_to_drop = ['ever_married', 'residence_type']

dict_null_value = {
    'gender': 'female',
    'ever_married': 'yes',
    'work_type': 'private',
    'residence_type': 'rural',
    'smoking_status': 'never smoked',
    'hypertension': 0.0,
    'heart_disease': 0.0,
    'age': 43.0,
    'avg_glucose_level': .039853,
    'bmi': 28.112721,
}

feature_mapping = {
    0: "Gender",
    1: "Hypertension",
    2: "Heart Disease", 
    3: "Age",
    4: "Average Glucose Level",
    5: "BMI",
    6: "Work Type - Private",
    7: "Work Type - Self-employed", 
    8: "Work Type - govt_job",
    9: "Work Type - Children",
    10: "Work Type - Never worked",
    11: "Smoking Status - Never smoked",
    12: "Smoking Status - Formerly smoked",
    13: "Smoking Status - Smokes",
    14: "Smoking Status - Unknown"
}

def generate_text(prompt):
    gemini = genai.GenerativeModel('gemini-pro')
    response = gemini.generate_content(prompt)
    return response.text

def preprocess_problem(instance, list_problems):
    rmp = []
    for problem in list_problems:
        cur_pro = problem.lower()
        if 'work' in cur_pro:
            if 'work' not in instance.columns:
                rmp.append(problem)
                continue
            tmp = ' '.join(cur_pro.split()[3:])
            if tmp != instance['work_type'].iloc[0]:
                rmp.append(problem)
                continue
        if 'smoking' in cur_pro:
            if 'smoking' not in instance.columns:
                rmp.append(problem)
                continue
            tmp = ' '.join(cur_pro.split()[3:])
            if tmp != instance['smoking'].iloc[0]:
                rmp.append(problem)
                continue
    for p in rmp:
        list_problems.remove(p)
    return list_problems

def get_advice_from_gemini(pred, instance, exp):
    feature_contributions = exp.local_exp[1]

    positive_feature_ids = [feature_mapping[feature_id] for feature_id, importance in feature_contributions if importance > 0]

    processed_problem = preprocess_problem(instance, positive_feature_ids)
    text = ""
    for problem in processed_problem:
        if 'BMI' in problem:
            text += '\n- ' + f'{problem}: {instance["bmi"].iloc[0]}'
        elif 'Glucose' in problem:
            text += '\n- ' + f'{problem}: {instance["avg_glucose_level"].iloc[0]}'
        elif ('Hypertension' or 'Heart') in problem:
            text += '\n- ' + f'Got {problem}'
        else:
            text += '\n- ' + problem
    final_text = (
        f"I am {instance['age'].iloc[0]} years old and I got predicted stroke with {pred[1]:.2f}%. Here is what model assumes that is relevant to my stroke prediction:\n"
        + text
        + "\nGive me advice for each one above to improve how to reduce stroke prediction ! Do not add additional information."
    )
    respone = generate_text(prompt=final_text)
    return respone

def preprocess_problem(instance, list_problems):
    rmp = []
    for problem in list_problems:
        cur_pro = problem.lower()
        if 'work' in cur_pro:
            if 'work' not in instance.columns:
                rmp.append(problem)
                continue
            tmp = ' '.join(cur_pro.split()[3:])
            if tmp != instance['work_type'].iloc[0]:
                rmp.append(problem)
                continue
        if 'smoking' in cur_pro:
            if 'smoking' not in instance.columns:
                rmp.append(problem)
                continue
            tmp = ' '.join(cur_pro.split()[3:])
            if tmp != instance['smoking'].iloc[0]:
                rmp.append(problem)
                continue
    for p in rmp:
        list_problems.remove(p)
    return list_problems

def process(user_info):
    
    assert len(user_info) == 1
    for col in  dict_null_value:
        if col in user_info and user_info[col].iloc[0] == None:
            user_info[col] = dict_null_value[col]

    format_info = format_form(user_info)
    x = format_info.drop(['stroke'], axis=1)
    y = format_info['stroke']
    x, y = get_pass_data(x, y)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    preprocessor = model['preprocessor']
    classifier = model['classifier']
    
    data_preprocessed = preprocessor.transform(x)


    train = pd.read_csv('train.csv')
    train_format = format_form(train)
    x_train = train_format.drop(['stroke'], axis=1)
    y_train = train_format['stroke']
    x_train, y_train = get_pass_data(x_train, y_train)

    x_train_transform = preprocessor.transform(x_train)
    explainer = LimeTabularExplainer(x_train_transform, mode="classification")

    exp = explainer.explain_instance(data_preprocessed.reshape(-1), classifier.predict_proba)
    pred_probs = exp.predict_proba

    advice = get_advice_from_gemini(pred_probs, x, exp)

    
    print("\nPredicted Class Probabilities:")
    print(pred_probs)
    return pred_probs, advice