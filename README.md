# 🧠 Stroke Analysis: A Data-Driven Exploration

## 🔍 What is it?
Stroke is one of the leading causes of mortality and disability worldwide. Our project applies data-driven techniques to analyze stroke risk factors and develop predictive models to assist individuals and healthcare professionals in mitigating stroke risks. We also built a Minimal Viable Product (MVP) that provides risk predictions, recommendations, and educational resources.

## 🎯 Why this project? 
The motivation behind this project stems from the critical need to identify and prevent strokes before they occur. By leveraging machine learning and exploratory data analysis, we aim to uncover key factors influencing stroke risks and provide actionable insights that can drive better health outcomes.

## 👥 Who can benefit? 
This project is designed for:
- 🏥 Healthcare professionals seeking data-driven insights into stroke risk factors.
- 📊 Researchers and data scientists exploring predictive analytics in medical applications.
- 🧑‍⚕️ Individuals who want to assess their stroke risk and take preventive measures.

## 🎥 Demo and Results
### 📈 Performance Metrics:
Our machine learning models achieved the following results:
- **✅ Accuracy:** 95.79%
- **📊 ROC-AUC Score:** 0.9003
- **🔢 Macro Average:** Precision = 0.7129, Recall = 0.5679, F1-Score = 0.5983
- **🔍 Weighted Average:** Precision = 0.9436, Recall = 0.9579, F1-Score = 0.9471

### 🖥 MVP Features:
- **🧠 Explainable Risk Prediction:** Users input health parameters, and the system predicts stroke risk with explanations.
- **💡 Recommendations:** Personalized suggestions to reduce stroke risk.
- **📚 Educational Platform:** A website with data visualizations and analyses for better awareness.

[🔗 GitHub Repository](https://github.com/QuanHoangNgoc/Stroke-Analysis-Data-Driven-Exploration)

## ⚙️ How did we do it?
### 1️⃣ Data Analysis & Feature Engineering:
- 🗂 Dataset sourced from Kaggle, containing **15,303 samples** with demographic, medical, and lifestyle attributes.
- 📊 Performed **Exploratory Data Analysis (EDA)** to identify key risk factors, using statistical tests and visualizations.
- 📉 Applied **Chi-squared tests, correlation analysis**, and **Partial Dependence Plots (PDP)** to validate relationships between features and stroke risk.

### 2️⃣ Model Development:
- 🤖 Used **XGBoost, CatBoost, and Random Forest** for classification.
- ⚖️ Balanced dataset and optimized model parameters.
- 🔄 Combined models with **Soft Voting** to improve prediction reliability.

### 3️⃣ Validation & Interpretability:
- ✅ Performed **Factor Evaluation** to assess feature importance.
- 📈 Used **Probability Validation** to confirm trends identified in EDA.
- 🔬 Ensured the model’s conclusions align with real-world stroke predictors.

## 📚 Key Learnings
- **📅 Age and glucose levels** are the most influential factors in stroke risk.
- **❤️ Medical history (hypertension & heart disease)** significantly increases stroke probability.
- **🚬 Lifestyle choices, such as smoking,** are critical determinants of stroke risk.
- **🏡 Marital status and geographic location** have minimal impact.
- **🧪 Combining EDA insights with machine learning validation** improves result reliability.

## 🏆 Achievements
- 🏅 Built an effective **stroke risk prediction model** with high accuracy.
- 🎨 Developed a **user-friendly MVP** that bridges AI and healthcare.
- 🔬 Provided **data-backed insights** for preventive healthcare measures.
- 📊 Achieved a **ROC-AUC score of 0.9003**, ensuring model robustness.

## 👨‍💻 Author - Support & Contributions
This project was developed by **Quan-Hoang-Ngoc** and Group 9 from **University of Information Technology (UIT), Ho Chi Minh City, Vietnam**.

If you found this project useful, consider supporting:
- 🌍 GitHub: [@QuanHoangNgoc](https://github.com/QuanHoangNgoc)
- ☕ Buy Me a Coffee: [Donate Here](https://www.buymeacoffee.com/QuanHoangNgoc)

Your support helps us continue research and development in data-driven healthcare solutions! 🚀

