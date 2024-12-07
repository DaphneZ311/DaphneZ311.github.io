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

## Preprocessing Steps
	•	Class Imbalance Handling: Applied SMOTE to address the imbalance in user vs. non-user categories.
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

<p>
When \(a \ne 0\), there are two solutions to \(ax^2 + bx + c = 0\) and they are
  \[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]
</p>

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From the results, XGBoost was the best-performing model, achieving the highest AUC-ROC and accuracy. The findings indicate that:
	1.	Personality Traits: Neuroticism, extraversion, and impulsivity strongly predict alcohol consumption.
	2.	Demographics: Age is a significant predictor, with younger individuals showing higher rates of usage.
	3.	Behavioral Patterns: Sensation seeking and impulsivity are shared traits across alcohol, nicotine, and cannabis users.

## Conclusion

This project demonstrates how machine learning can predict substance use based on personality traits and demographics. The insights gained can inform public health initiatives and help target interventions more effectively.

Here is how this work could be developed further in a future project.

	1.	Investigate the interaction between personality traits and demographic factors.
	2.	Expand the study to include other substances like cocaine or heroin.
	3.	Explore deep learning methods for potential performance gains.

## References
[1] UCI Machine Learning Repository: Drug Consumption (Quantified).

[back](./)
