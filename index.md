<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## My Project

I applied machine learning techniques to investigate the relationships between alcohol, nicotine, and cannabis consumption based on personality traits and demographic data. Below is my report.

***

## Introduction 

Substance use, particularly alcohol, nicotine, and cannabis, has significant implications for public health. Understanding the factors associated with their consumption can provide insights for preventive measures and interventions.

The dataset used in this study, sourced from the UCI Machine Learning Repository, contains demographic information and personality traits of individuals, along with their substance use behaviors. This allows us to apply machine learning techniques to predict substance use and uncover patterns in behavior.

By using supervised learning models such as Random Forest, Logistic Regression, and XGBoost, I aimed to classify individuals as users or non-users for each substance and identify key predictors of substance use. The findings demonstrate how demographic and psychological factors influence substance use patterns.

## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

The dataset contains 1885 instances and includes features such as:

	•	Demographics: Age, gender, education level, country, and ethnicity.
	•	Personality Traits: Neuroticism, extraversion, openness, agreeableness, and conscientiousness (measured using NEO-FFI-R).
	•	Behavioral Traits: Impulsivity and sensation seeking.

The target variables are binary indicators of alcohol, nicotine, and cannabis use:

	•	0: Non-user.
	•	1: User.

![image](https://github.com/user-attachments/assets/f990d918-14c6-473f-acb2-bf598865b930)

*Figure 1: Alcohol consumption distribution bar plot.*

![image](https://github.com/user-attachments/assets/2578c5fe-cae9-4a4a-8104-905158d1c8f5)

*Figure 2: Cannabis consumption distribution bar plot.*

![image](https://github.com/user-attachments/assets/2450d3a1-10e5-486d-93fb-3948f8690a94)

*Figure 3: Nicotine consumption distribution bar plot.*

![image](https://github.com/user-attachments/assets/7ed89f42-020e-4f4b-9ce6-3dc673f42aa6)

*Figure 4: nscore distribution plot.*

## Data Visualization 

These heatmaps illustrate the percentage overlap between usage of alcohol, nicotine, and cannabis, based on the raw unbalanced dataset. Each heatmap compares two substances, breaking down the percentages of users (1) and non-users (0) for both categories. The findings are as follows:

![image](https://github.com/user-attachments/assets/bca0639c-94a5-4b79-8764-c762299f321a)
Among individuals who do not consume alcohol (0), 58.1% also do not use nicotine, while 41.9% do. For alcohol users (1), the distribution is nearly balanced: 57.3% also use nicotine, while 42.7% do not. This suggests a moderate overlap between alcohol and nicotine use.

![image](https://github.com/user-attachments/assets/f639b3e1-2325-4c36-aa69-0f278d88bd4f)
For non-drinkers (0), 61.0% also do not use cannabis, while 39.0% are cannabis users. Among drinkers (1), the distribution is again closer to balance: 54.1% are also cannabis users, and 45.9% are not. Alcohol and cannabis usage show a weaker overlap compared to alcohol and nicotine.

![image](https://github.com/user-attachments/assets/94e1bc16-daa5-4804-92e6-7aa58643bd57)
Among non-cannabis users (0), 71.0% do not use nicotine, while 29.0% do. Cannabis users (1) show a strong overlap with nicotine users, with 80.4% also using nicotine and only 19.6% not using nicotine. This indicates a significant relationship between cannabis and nicotine use.

One limitation is that the data here is unbalanced, especially for alcohole consumption. This is why preprocessing is important. 


## Preprocessing Steps
	•	Class Imbalance Handling: 
 Applied SMOTE to address the imbalance in user vs. non-user categories.
 
 Code: 

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X, y1 = smote.fit_resample(X, y1)
```

	•	Feature Scaling: Standardized numeric features to improve model performance.
	•	Target Binarization: Combined usage levels into two categories (0 for non-user, 1 for user).

## Modelling

Supervised learning methods were employed, including:

	1.	Random Forest: To establish a baseline and identify feature importance.
	2.	Logistic Regression: For interpretable predictions based on linear relationships.
	3.	XGBoost: To capture complex, non-linear patterns in the data.

The models were evaluated using:

	•	Accuracy: To measure overall prediction correctness.
	•	AUC-ROC: To assess discriminatory power between users and non-users.
	•	Classification Report: To evaluate precision, recall, and F1-score for each class.

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}
```


## Results

Alcohol Consumption

	The machine learning models trained to predict alcohol consumption based on demographic and personality traits yielded diverse performance results. The Random Forest model achieved the highest accuracy (97.1%) and AUC-ROC (0.995), demonstrating strong discriminatory power and balanced precision-recall performance for both user and non-user classes. XGBoost also performed exceptionally well, with an accuracy of 95.7% and an AUC-ROC of 0.985, showcasing its capability to handle complex patterns in the data. 
	In contrast, Logistic Regression underperformed significantly, achieving an accuracy of 63.7% and an AUC-ROC of 0.672, suggesting that linear relationships were insufficient to capture the intricacies of the dataset. These results highlight the importance of using non-linear models like Random Forest and XGBoost for capturing the complex interactions between features in predicting alcohol consumption.


Nicotine Consumption

	The predictive models for nicotine consumption demonstrated moderate performance. XGBoost achieved the highest accuracy (70.3%) and a decent AUC-ROC of 0.776, Random Forest followed closely with an accuracy of 69.3% and a slightly higher AUC-ROC of 0.788, Logistic Regression, while comparable in accuracy (69.1%), had the lowest AUC-ROC (0.750), suggesting limited capability to capture non-linear relationships in the dataset. 

Cannabis Consumption

	The predictive models for cannabis consumption achieved consistent and moderate performance across all tested algorithms. Random Forest had an accuracy of 78.8% and the highest AUC-ROC of 0.869, Logistic Regression performed slightly lower with an accuracy of 77.5% and an AUC-ROC of 0.865, XGBoost also delivered robust results with an accuracy of 79.0% and an AUC-ROC of 0.855, demonstrating effective handling of complex relationships. Overall, all three models showed comparable performance.


## Feature Selection

Feature importance analysis was conducted using the Random Forest model to identify the most influential features in predicting substance use (alcohol, nicotine, and cannabis). The feature importance scores, derived from the model, indicate how much each feature contributes to the prediction.

Code:

```python
feature_importances = rf_model.feature_importances_
```
![image](https://github.com/user-attachments/assets/baa221d7-1e3e-484b-8e88-e5eec4d4d41b)

Key findings include:

	Top Features: Traits such as Nscore (Neuroticism), Oscore (Openness), Ascore (Agreeableness), Escore (Extraversion), Cscore (Conscientiousness) consistently ranked as the most impactful features across all predictions.
 
## Results 

After applying feature selection, the predictive models performance does not change significantly. 
For alcohol, Random Forest and XGBoost achieved high accuracies of 94.7% and 94.4%, respectively, with strong AUC-ROC values (0.990 and 0.984), while Logistic Regression underperformed with 53.7% accuracy. For cannabis, Random Forest outperformed Logistic Regression and XGBoost, with 69% accuracy and a 0.757 AUC-ROC. Similarly, for nicotine, Random Forest achieved the best results (69% accuracy, 0.754 AUC-ROC). Feature selection consistently improved model interpretability and highlighted key predictors, though non-linear models such as Random Forest and XGBoost performed significantly better overall.


## Discussion

The analysis highlights the effectiveness of machine learning models in predicting alcohol, cannabis, and nicotine consumption based on personality traits and demographic data. Across all substances, Random Forest and XGBoost consistently outperformed Logistic Regression, demonstrating the importance of non-linear models in capturing complex relationships within the dataset. For alcohol consumption, the models achieved exceptionally high accuracy and AUC-ROC values, showcasing strong predictive power and reliability.

In contrast, predictions for cannabis and nicotine were moderately accurate, with Random Forest performing slightly better than XGBoost and Logistic Regression. This suggests that the patterns in cannabis and nicotine use are more complex or less strongly correlated with the available features. Feature selection played a significant role in improving model interpretability, confirming the critical influence of traits such as conscientiousness, neuroticism, and impulsivity.


## Conclusion

This project demonstrates how machine learning can predict substance use based on personality traits and demographics. The insights gained can inform public health initiatives and help target interventions more effectively.

Here is how this work could be developed further in a future project.

	1.	Investigate the interaction between personality traits and demographic factors.
	2.	Expand the study to include other substances like cocaine or heroin.
	3.	Explore deep learning methods for potential performance gains.

## References
[1] UCI Machine Learning Repository: Drug Consumption (Quantified).

[back](./)
