# A Comprehensive Machine Learning Framework for Heart Disease Classification Using Multi-Feature Clinical Data (Dec 2025)


# ABSTRACT 
Heart disease remains one of the leading causes of mortality worldwide, and the growing availability of clinical data provides an opportunity to enhance diagnostic support using machine learning techniques. This study presents a comprehensive analytical framework designed to classify heart disease outcomes using a dataset of 920 patient records containing fifteen clinical and diagnostic variables. The framework integrates traditional baselines, ensemble learning models, and modern deep learning architectures to deliver a balanced and methodologically robust evaluation. Logistic Regression, SVM, and Random Forest were used as foundational models, while XGBoost, LightGBM, and CatBoost formed the core gradient boosting suite. To extend the analysis beyond predictable approaches, the study incorporates TabNet and a shallow neural network as deep learning components. Model performance was evaluated using standard metrics such as accuracy, F1-score, precision, recall, and ROC-AUC. Gradient boosting methods consistently achieved the strongest performance, with CatBoost emerging as the most reliable classifier across all evaluation criteria. Feature importance analysis and SHAP values were employed to interpret model behaviour and highlight the contribution of critical variables such as age, chest pain type, cholesterol level, and maximum heart rate. The results demonstrate that advanced machine learning models can meaningfully support clinical decision making and offer insights that complement traditional diagnostic methods. The framework developed here provides a practical foundation for future medical predictive analytics research using structured clinical datasets.
INDEX TERMS Heart disease classification, machine learning, ensemble learning, interpretable artificial intelligence, clinical decision support.    

 
# I. INTRODUCTION
This Heart disease continues to be a major public health challenge, contributing significantly to mortality and long-term illness across the globe. Early identification of individuals at heightened risk is central to improving outcomes, yet the clinical indicators associated with heart disease development often interact in complex ways that can be difficult to interpret through conventional statistical approaches alone. According to long standing epidemiological evidence, cardiovascular diseases continue to rank among the leading causes of death worldwide (World Health Organization, 2021; Benjamin et al., 2019). With the growth of accessible clinical datasets and the increasing application of machine learning techniques, there is strong potential to support clinicians by offering more nuanced, data driven risk assessments.
Conventional statistical tools, such as Logistic Regression and risk-scoring systems, have long been used to support cardiovascular assessment (D’Agostino et al., 2008). While these approaches are valued for their interpretability, they are limited in their ability to capture non-linear relationships, higher-order interactions, and subtle diagnostic patterns embedded within structured clinical datasets. In recent years, machine-learning techniques have shown promising advances in medical classification tasks, offering greater flexibility and enhanced modelling capacity (Shahid et al., 2020; Beam & Kohane, 2018).
Conventional statistical tools, such as Logistic Regression and risk scoring systems, have long been used to support cardiovascular assessment (D’Agostino et al., 2008). While these approaches are valued for their interpretability, they are limited in their ability to capture non-linear relationships, higher order interactions, and subtle diagnostic patterns embedded within structured clinical datasets. In recent years, machine learning techniques have shown promising advances in medical classification tasks, offering greater flexibility and enhanced modelling capacity (Shahid et al., 2020; Beam & Kohane, 2018).
Among these techniques, ensemble learning and gradient boosting algorithms, such as XGBoost, LightGBM, and CatBoost, have demonstrated strong performance on tabular healthcare datasets due to their capacity to model complex interactions while maintaining robustness to noise and multicollinearity (Chen & Guestrin, 2016; Ke et al., 2017; Dorogush et al., 2018). Deep learning models designed specifically for highly structured data, including TabNet and modern neural architectures, have further expanded the landscape by providing representation learning capabilities that complement ensemble methods (Arik & Pfister, 2021).
This study builds upon these developments by applying a comprehensive machine learning framework to a dataset of 920 patient records, each containing fifteen clinical and diagnostic attributes commonly used in heart disease screening. The modelling strategy integrates traditional baselines, powerful boosting algorithms, and deep learning components to provide a balanced evaluation of predictive performance across multiple methodological families.

Model interpretability is also central to this analysis. Recent discussions emphasise the importance of transparency and clinical explainability in AI driven decision support (Doshi-Velez & Kim, 2017; Lundberg & Lee, 2017). Techniques such as feature importance analysis and SHAP (Shapley additive explanations) values help reveal the contribution of individual predictors, providing a clearer understanding of how variables such as age, chest pain type, cholesterol levels, and exercise induced responses influence diagnostic outcomes.
This work contributes to the growing body of research applying machine learning in cardiovascular diagnosis by developing a structured, interpretable, and methodologically robust framework for heart-disease classification. The findings highlight both the predictive value and practical significance of modern algorithms when applied responsibly to structured clinical data.


# II. RELATED WORKS
Research on heart disease prediction has gained significant attention in the past two decades, particularly as machine learning techniques have advanced and become more accessible to healthcare researchers. Early studies relied primarily on traditional statistical methods, with Logistic Regression and decision rule systems forming the basis of cardiovascular risk assessment. One of the most influential examples is the Framingham Risk Score, which used multivariate regression to estimate long term cardiovascular risk (D’Agostino et al., 2008). Although these methods provided foundational insights, they were constrained by their linear assumptions and limited capacity to capture complex interactions among clinical variables.

As digital health records and public datasets became increasingly available, researchers began to experiment with classical machine learning techniques. Support Vector Machines, k-Nearest Neighbours, Decision Trees, and Naïve Bayes have all been applied to heart disease classification tasks, often demonstrating improved performance over purely statistical approaches (Nahar et al., 2013; Son et al., 2012). Random Forest, in particular, became a popular choice due to its robustness and ability to model non-linear relationships (Kumari & Chitra, 2013). Although these algorithms offered measurable gains, their accuracy was still limited by the relatively simple decision boundaries they imposed.

The emergence of ensemble learning methods marked a substantial shift in predictive modelling. Gradient boosting frameworks such as XGBoost, LightGBM, and CatBoost have consistently achieved state of the art results on structured tabular data, including cardiovascular datasets (Chen & Guestrin, 2016; Ke et al., 2017; Dorogush et al., 2018). These models combine multiple weak learners to build more powerful classifiers, enabling richer interactions among features such as cholesterol level, blood pressure, chest pain type, and exercise induced angina. Studies comparing gradient boosting with classical techniques frequently report significant improvements in accuracy, sensitivity, and AUC values, making boosting methods central to modern clinical prediction research (Shahid et al., 2020).

More recently, deep learning approaches have been explored for heart disease prediction. While deep neural networks have transformed fields such as imaging and speech recognition, their application to structured clinical data remains challenging due to small dataset sizes and limited feature dimensionality. Still, shallow neural architectures have been used to capture non-linear patterns in cardiovascular datasets with promising but inconsistent results (Khan et al., 2020). The development of models tailored to structured data inputs, such as TabNet, has further expanded the deep learning landscape by introducing attention based mechanisms designed specifically for structured datasets (Arik & Pfister, 2021). These newer models have shown potential but are not yet extensively validated in medical data such as heart disease dataset.
Despite these advances, several important gaps remain in the existing literature. A common limitation is the narrow focus on a single algorithm, which makes it difficult to evaluate how different modelling families perform relative to one another. Many studies rely on small subsets of the UCI heart disease dataset and restrict their analysis to only one or two models, limiting the generalizability of the findings (Son et al., 2012; Nahar et al., 2013). Furthermore, interpretability is often treated as secondary, even though transparency is essential when ML systems are designed to support clinical decision making. Few papers provide a thorough explanation of model behaviour using techniques such as SHAP values, leaving a gap in understanding of how key clinical variables influence predictions.

Another important gap lies in the integration of modern boosting algorithms with deep learning models within a unified comparative framework. Although ensemble methods and neural networks have each been explored independently, studies rarely evaluate them together in a systematic manner on the same dataset. This limits the field’s ability to draw meaningful conclusions about which algorithms offer the best balance between accuracy, generalizability, and interpretability in small to medium sized clinical datasets.

This study addresses these gaps by presenting a comprehensive machine learning framework that evaluates baseline models, state of the art boosting algorithms, and deep learning architectures on a unified heart disease dataset containing 920 patient records and fifteen clinical variables. By incorporating Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, CatBoost, TabNet, and a shallow neural network, the study provides one of the most extensive comparative analyses on structured cardiovascular data. The work also integrates interpretability through feature importance measures and SHAP values, offering clear insights into the contributions of key predictors such as chest pain type, cholesterol level, age, and maximum heart rate. This emphasis on transparency strengthens the clinical relevance of the findings and supports the responsible use of machine learning in diagnostic settings. Thus, the study contributes to the field by combining methodological breadth, rigorous evaluation, and interpretability, helping to bridge an important gap in the literature on ML based heart disease prediction. 

# III. METHEDOLOGY
This study follows the CRISP–DM framework, a well-established methodology for data analytics research that supports systematic planning and transparent reporting (Shearer, 2000). The framework includes six phases: Business Understanding, Data Understanding, Data Preparation, Modelling, Evaluation, and Deployment. For academic purposes, the final phase is framed as “Knowledge Deployment,” as it relates to insights rather than operational implementation. Figure 1 presents the overall workflow used in this research.

<img width="241" height="173" alt="image" src="https://github.com/user-attachments/assets/9f00cd85-b85d-41c2-973f-88d48fb6767c" />





FIGURE 1. The CRISP–DM workflow applied in this study

# A. BUSINESS UNDERSTANDING
The primary aim of this study is to develop a robust machine learning framework to classify the presence of heart disease using fifteen clinical variables for 920 patients. The study seeks to answer three central questions:
1)	Which machine learning models achieve the highest predictive performance on heart disease data?
2)	How do advanced gradient boosting models compare to deep learning and traditional baselines?
3)	Which clinical features contribute most strongly to predictive accuracy, and how can model behaviour be interpreted?
Improving predictive accuracy and interpretability is essential for supporting clinicians, especially in environments where early detection can significantly reduce mortality.

# B. DATA UNDERSTANDING
This data related to Cleveland heart disease dataset which is available on UCI Machine Learning Repository. The dataset involving a variety of separate mathematical or statistical variables, multivariate numerical data analysis. The dataset consists of 920 patient records and 15 variables, including demographic characteristics, blood pressure metrics, cholesterol levels, ECG results, exercise responses, and angiographic outcomes. Table 1 summarizes the variables used in this analysis.

TABLE 1. Summary of Heart Disease Variables 
_________________________________________
Variable        	Description
id	              Unique patient identifier
age	              Age in years
sex	              Biological sex (0 = female, 1 = male)
cp	              Chest pain type
trestbps	        Resting blood pressure
chol	            Serum cholesterol (mg/dl)
fbs	              Fasting blood sugar (>120 mg/dl)
restecg	          Resting electrocardiographic results
thalch	          Maximum heart rate achieved
exang	           Exercise-induced angina
oldpeak	         ST depression induced by exercise
slope	           Slope of peak exercise ST segment
ca	             Number of major vessels coloured by fluoroscopy
thal	           Thalassemia status
num	             Outcome variable (heart disease severity)



# C. DATA PREPARATION CLEANING 
The data preparation stage involved several sequential steps designed to ensure data quality, standardize feature representation, and support reliable model performance. These steps included data cleaning, feature encoding, feature scaling, and the creation of training and testing subsets. The dataset was first examined for invalid or inconsistent entries, including incorrect or negative values in clinical variables. Missing values were then addressed using appropriate imputation methods. Categorical features such as thal and ca were imputed using their respective mode values to maintain the integrity of category distributions. Continuous variables with missing values were imputed using the median of each feature, an approach that preserves the underlying distribution and reduces the influence of extreme values.

# D. FEATURE ENCODING AND SCALING
Categorical variables, including cp, thal, slope, and other non-numeric attributes, were converted into numerical representations through one hot encoding for the majority of models. This ensured compatibility with algorithms that require numerical input. An exception was made for CatBoost, which incorporates native handling of categorical data through target-based encoding, eliminating the need for one hot transformation and improving efficiency and performance (Dorogush et al., 2018).
To ensure comparable feature magnitudes, models sensitive to feature scaling specifically Support Vector Machines, Logistic Regression, and deep neural networks were trained using data normalized with StandardScaler. This transformation standardized numerical variables to zero mean and unit variance. In contrast, tree-based models such as Random Forest and gradient boosting algorithms were trained on unscaled data, as they rely on decision rules that are unaffected by feature magnitude.

# E. TRAIN AND TEST SPLIT 
The dataset was split into 80% for training and 20% for testing datasets while preserving the original class distribution of the target variable. To enhance the robustness and generalizability of model estimates, 10-fold cross validation was employed during the training phase for all models. This ensured that performance metrics were not overly dependent on a single partition of the data and provided a more comprehensive evaluation of model stability.

# F. MODELING 
A multi-model framework was employed to provide methodological breadth and enable a balanced comparison across statistical, ensemble, and deep learning approaches. This structure ensured that the strengths and limitations of each modelling paradigm could be evaluated within a consistent experimental pipeline.

# G. BASELINE MODELS 
The baseline models served as fundamental reference points against which the performance of more advanced algorithms could be compared. The following algorithms was used include Logistic Regression which is a linear classification method that models the log odds of the target variable and provides interpretable coefficient estimates. It is widely used in medical research due to its simplicity and transparency. Support Vector Machine (RBF Kernel) constructs an optimal separating hyperplane in a high dimensional space, using a radial basis function to capture non-linear boundaries between classes. Its robustness to overfitting makes it suitable for medium-sized clinical datasets. Random Forest is an ensemble of decision trees built on random subsets of data and features, reducing variance and improving generalization. It effectively models complex interactions while maintaining resilience to noise (Kumari & Chitra, 2013).

# H. GRADIENT BOOSTING CORE MODELS
Gradient boosting algorithms were included due to their strong performance on structured datasets and their ability to model non-linear relationships. Where, XGBoost (Chen & Guestrin, 2016) is an efficient gradient boosted tree model that uses second order optimization, regularization, and subsampling to prevent overfitting. LightGBM (Ke et al., 2017) is designed for high-speed learning, using histogram  splitting and leaf wise tree growth to handle large feature spaces with minimal computational cost. CatBoost (Dorogush et al., 2018) introduces ordered boosting and native handling of categorical variables, reducing prediction shift and improving accuracy without extensive preprocessing. These boosting models are known for their ability to automatically capture feature interactions, perform implicit variable selection, and deliver state-of-the-art results on structured data.

# I. DEEP LEARNING COMPENT 
To enhance methodological diversity, the study incorporated two deep learning architectures (TabNet and shallow neural network) tailored for structured clinical data. TabNet (Arik & Pfister, 2021) is an attention-based model that learns sequential feature representations, enabling understanding and sparse decision paths. Its design allows it to selectively focus on the most informative attributes at each step of the learning process. A shallow feedforward neural network was also implemented, consisting of an input layer for the dataset features, two hidden layers (e.g., 64 and 32 units), ReLU activation, dropout regularization, and the Adam optimizer. This architecture provides a flexible non-linear model while reducing the risk of overfitting typically associated with deeper neural networks. Including deep learning methods allowed for a meaningful comparison between modern representation learning techniques and traditional ensemble learners on structured dataset.

# J. HYPERPARMETER TUNING
Hyperparameter optimization was conducted to enhance model performance and prevent overfitting. A hybrid strategy combining Grid Search and Bayesian optimization was employed, allowing both in-depth evaluation of selected parameter ranges and adaptive search of promise regions within the parameter space. For the boost models, tuning focused on the learning rate, tree depth, and subsample parameters, which control model complexity and regularization. The SVM model required optimization of the C parameter and gamma, both of which influence the smoothness of the decision boundary. For Random Forest, the optimization targeted the number of estimators and maximum tree depth, balancing predictive power with generalization. Deep learning models, including TabNet and the shallow neural network, were tuned by adjusting the learning rate, batch size, dropout rate, and number of training epochs. All tuning procedures were set within a 10-fold cross validation framework to ensure stable and generalized hyperparameter selection across the dataset.

# K. IMPLEMENTION DETAILS 
All analyses were implemented in Python using the scikit-learn, XGBoost, LightGBM, and CatBoost libraries. The heart disease dataset was first loaded and inspected to verify structure, variable types, and target distribution. A binary outcome variable was derived from the original multi-class num attribute, distinguishing between the presence and absence of heart disease. Data preparation was conducted within a unified pipeline comprising median imputation for numerical variables, mode imputation for categorical features, one-hot encoding of non-numeric attributes, and tuning of continuous predictors for scale sensitive models. The dataset was then split into training and testing subsets using an 80/20 stratified split to preserve class balance. Each classifier was embedded in the same preprocessing pipeline to ensure fair comparison, and model performance was evaluated on the held-out test set using accuracy, precision, recall, F1-score, and ROC–AUC. For the best performing tree-based model, SHAP analysis was applied to generate feature attribution explanations and support interpretability.

# L.  EVALUATION
Model performance was assessed using a meaningful evaluation metrics that collectively capture predictive accuracy, error distribution, and discriminative capability. These metrics included accuracy, precision, recall, F1-score, and ROC and AUC, enabling a multi-dimensional assessment of classification performance. Accuracy provided an overall measure of correct predictions, while precision and recall quantified the model’s ability to correctly identify positive cases without overpredicting. The F1-score served as a harmonic balance between precision and recall, particularly relevant in datasets with asymmetric class distributions. In addition, ROC and AUC was used to evaluate the model’s capacity to distinguish between patients with and without heart disease across varying probability thresholds. Confusion matrices were generated to visualize misclassification patterns and highlight potential model biases. ROC curves were plotted for all models to facilitate direct graphical comparisons of discriminative performance.

# M. INTERPRETABILIY 
Interpretability was corporate into our approach to make sure the model was transparent and could genuinely support clinical decision-making. Using feature importance analysis from the Random Forest model, we identified the key predictors that were driving the results. Cholesterol turned out to be the most influential feature, closely followed by age and maximum heart rate (thalch), showing that both metabolic health and cardiovascular function matter. Other strong contributors were ST depression (oldpeak) and chest pain type (cp), which makes sense given their well-known link to myocardial ischemia. Exercise related attributes especially maximum heart rate (thalch), ST depression (oldpeak), and exercise induced angina (exang) were especially significant, highlighting how the heart responds under physical stress as a key factor in the model's predictions. Demographic details like sex and resting ECG results added further nuance, though their impact was smaller. Looking deeper with SHAP interaction analysis, we saw that age consistently raised the predicted risk of disease, while resting blood pressure (trestbps) showed only a weak interaction with itself. The relationship between age and resting blood pressure was modest, meaning these factors mainly contribute independently rather than amplifying each other. Altogether, these insights help confirm that the model's reasoning is not only consistent with medical knowledge but also interpretable enough to be useful in real world screening or diagnostic support.


# N.  KNOWLEDGE DEPLOYMENT
Although the study does not focus on real time clinical deployment, the final phase emphasizes the translation of analytical findings into actionable knowledge. The insights generated through the modelling and interpretability steps provide valuable guidance for the development of clinical decision support systems, offering a data driven foundation for improved cardiovascular screening and risk assessment. The results also update the potential integration of predictive algorithms into cardiology workflows, supporting clinicians in early detection and patient triage. Additionally, the findings highlight roads for future research, including multimodal modelling, cross institutional validation, and ethical evaluation of AI driven diagnostics. The framework developed in this study serves as a replicable and scalable blueprint for applying advanced machine learning techniques to structured health datasets	

# O. HEART DISEASE DATA ANALYSIS 
This section presents a detailed analysis of the heart disease dataset to understand its structure, distribution, and key clinical characteristics. The analysis begins with exploratory data assessment, focusing on demographic trends, physiological measurements, and diagnostic indicators. Particular attention is given to identifying missing values, variable relationships, and potential patterns associated with heart disease outcomes. Visual and statistical summaries are used to highlight notable differences between patients with and without diagnosed heart disease. These insights provide the foundation for subsequent modelling and interpretation within the study.


# IV. RESULTS  
This section presents the analytical findings derived from the heart-disease dataset, encompassing exploratory data analysis, predictive modelling, and model interpretability. The analysis begins by examining the distribution and relationships of key clinical variables to identify patterns associated with heart-disease outcomes. Subsequently, the predictive performance of multiple machine-learning models is evaluated and compared using clinically relevant metrics. Model outputs are further examined through confusion matrices and ROC curves to assess classification reliability. Finally, interpretability analyses are presented to explain model behaviour and highlight the most influential clinical predictors.

# A. EXPLORATORY DATA ANALYSIS
Figure 2. Binary distribution of heart disease cases in the dataset. This bar chart illustrates the proportion of patients without heart disease (target = 0) and with heart disease (target = 1) after converting the original multi-class variable into a binary outcome. The binary target distribution showed that 55.4% of patients were diagnosed with heart disease, while 44.6% had no disease, indicating only mild class imbalance.

<img width="214" height="180" alt="image" src="https://github.com/user-attachments/assets/2e243ba6-8aaa-4c3e-b5f4-ad7c2dd7a193" />


FIGURE 2. Binary distribution of heart disease

Histograms Figure 3. illustrate the distributions of the six primary numeric variables: age, resting blood pressure (trestbps), serum cholesterol (chol), maximum heart rate achieved (thalch), ST depression (oldpeak), and number of major vessels (ca). Distribution analysis of numeric variables revealed consistent patterns: age followed a near normal distribution centred around 50 and 60 years; resting blood pressure (trestbps) and cholesterol (chol) exhibited mild right-skewness; and maximum heart rate (thalch) showed a symmetric distribution with lower values prevalent among disease patients. ST depression (oldpeak) was heavily zero inflated, with higher values strongly associated with heart disease.

<img width="241" height="221" alt="image" src="https://github.com/user-attachments/assets/98441675-bf49-45bd-9b92-f070f4d96ee7" />

FIGURE 3. illustrate the distributions of the six primary numeric variables

Figure 4. Boxplots of numeric clinical features stratified by heart disease status. The figure compares distributions of age, resting blood pressure (trestbps), cholesterol (chol), maximum heart rate achieved (thalch), and ST depression (oldpeak) across patients with (target = 1) and without heart disease (target = 0). As shown the boxplots comparing features across the two target groups demonstrated clear discriminative patterns. Patients with heart disease were generally older, exhibited lower maximum heart rate during exercise, and displayed significantly elevated ST depression. Cholesterol and resting blood pressure displayed greater variability and weaker differentiation. These relationships were supported by the correlation analysis, where number of vessels (ca), oldpeak, thalch, and age showed the strongest correlations with the target.

<img width="242" height="183" alt="image" src="https://github.com/user-attachments/assets/a4c21e4f-66e1-407c-9bc0-3905d76e3098" />

FIGURE 4. Boxplots of numeric clinical features

Figure 5. Correlation heatmap of numeric clinical features and the binary target variable.  The figure displays pairwise Pearson correlation coefficients between numeric predictors and the heart-disease outcome (target). Warmer colors indicate positive correlations, while cooler colors indicate negative associations. As presents in correlation heatmap for the numeric predictors and the binary heart disease outcome. The strongest positive correlations with heart-disease presence were observed for the number of major vessels (ca; r = 0.46) and ST depression (oldpeak; r = 0.39), both of which reflect well-established clinical indicators of ischemic heart disease. Maximum heart rate (thalch) showed a moderately strong negative correlation (r = –0.39), indicating that lower exercise capacity is associated with higher disease likelihood. Age displayed a modest positive correlation (r = 0.28), while resting blood pressure and cholesterol demonstrated weak associations with the target. These results highlight that exercise-related and fluoroscopic variables carry the highest predictive value, reinforcing the non-linear and multi-factorial nature of the dataset.

<img width="222" height="197" alt="image" src="https://github.com/user-attachments/assets/4e8e6fbc-0e3c-4d61-8e4a-b3760fe195cb" />

FIGURE 5. Correlation heatmap of numeric clinical features and the binary target variable

B. MODEL PERFORMANCE
Nine machine learning models were evaluated, enclose baseline classifiers, ensemble methods, and a shallow neural network. Performance was assessed using accuracy, precision, recall, F1-score, and ROC-AUC as shown in Table 2. Tree based ensemble models consistently outperformed linear and deep models. Random Forest achieved the highest overall performance with an accuracy of 0.86 and an ROC and AUC of 0.92. The SVM (RBF) model demonstrated the highest recall (0.92), emphasizing strong sensitivity for clinical screening. Gradient Boosting, CatBoost, XGBoost, and LightGBM also exhibited high discriminative performance, with ROC and AUC values exceeding 0.89. The shallow neural network demonstrated lower performance metrics, consistent with expectations for medium sized structured datasets. Overall, the results highlight the superiority of ensemble tree methods for heart-disease prediction and reinforce their suitability for structured clinical data.

TABLE 2. illustrate the result of evaluated nine machine learning
  Model 	       Accu 	Precision 	Recall 	F1-sco 	ROC-AUC
2.Random F 	     0.86 	0.86       	0.90 	  0.88	  0.92
1.SVM 	         0.84 	0.82 	      0.92 	  0.87 	  0.92
3.Gradient B 	   0.86 	0.85 	      0.90 	  0.88	  0.91
7. CatBoost 	   0.85	  0.84        0.89 	  0.87 	  0.91
5. XGBoost 	     0.84	  0.84	      0.88	  0.86 	  0.91
0.Logistic Reg 	 0.84 	0.84	      0.88 	  0.86 	  0.90
4.HistGradientB  0.84 	0.83 	      0.89 	  0.86   	0.90
6.LightGBM 	     0.85	  0.84 	      0.90 	  0.87 	  0.89
8.Shallow NN	   0.80	  0.79	      0.87	  0.83	  0.86


Figure 6. displays the ROC curve for the Random Forest classifier. The model achieved an AUC of 0.92, demonstrating excellent discriminative ability. The ROC curve rises steeply toward the upper-left corner, indicating that the classifier maintains high sensitivity even at low false-positive rates. This behaviour is consistent with the model’s strong recall performance and its ability to correctly identify the majority of heart disease cases. The smooth convex shape of the curve reflects stable performance across threshold settings, further supporting the Random Forest model as the most effective and clinically reliable classifier among those evaluated.

<img width="241" height="155" alt="image" src="https://github.com/user-attachments/assets/c0932efa-6be3-4f7e-aa9e-6205cf231650" />

FIGURE 6. displays the ROC curve for the Random Forest classifier

Figure 7. presents the SHAP interaction values for age and resting blood pressure (trestbps). The plot shows that age exhibits a clear positive contribution to heart-disease prediction, with higher ages associated with higher SHAP interaction values. Resting blood pressure, in contrast, displays minimal interaction effects, with SHAP values concentrated around zero. The interaction between age and trestbps is modest, indicating that the model treats these predictors largely as independent contributors rather than synergistic risk factors. These findings reinforce earlier results showing that age provides meaningful diagnostic information, while resting blood pressure plays a comparatively limited role in model predictions.

<img width="241" height="178" alt="image" src="https://github.com/user-attachments/assets/2708e2cb-e262-4976-be72-24df611f11ce" />

FIGURE 7. presents the SHAP interaction values

# C.  FEATURE IMPORTANCE AND INTERACTION
Figure 7. presents the top 15 predictors ranked by feature importance in the Random Forest classifier. Cholesterol emerged as the most influential variable, followed by age and maximum heart rate (thalch), indicating that metabolic and exercise related indicators are central to the model’s decision making process. Chest pain of asymptomatic type and ST depression (oldpeak) also featured prominently, consistent with their clinical relevance in detecting ischemic changes. Resting blood pressure, exercise induced angina (exang), and atypical angina contributed moderate predictive value, while demographic and ECG related features provided smaller but meaningful contributions. These results demonstrate that the Random Forest model relies heavily on functional and physiological markers of cardiac stress, reflecting established clinical patterns in heart disease assessment.

<img width="215" height="177" alt="image" src="https://github.com/user-attachments/assets/0f325043-6b94-40ca-8c76-61fd20693c2e" />

Figure 8. presents the top 15 predictors ranked by feature importance


 V. DISCUSSION 
Please This study shows how a well-designed machine learning framework can classify heart disease using a varied, multi-feature dataset. Among the models we tested, ensemble tree methods especially Random Forest which performed best. Their strength comes from an ability to model complex, non-linear relationships, handle different types of data, and pick up on subtle interactions between features. The high ROC–AUC scores we saw across models confirm they could be useful in clinical settings, both for risk stratification and in pre-diagnostic screening.
Interpretability analysis also confirmed that the model makes sense clinically. Key predictors included factors tied to exercise tolerance like heart rate during exercise (thalch), exercise-induced ST depression (oldpeak), and exercise-induced angina (exang) along with metabolic markers such as cholesterol. These align with what we already know from cardiovascular research (Goff et al., 2014; Stone et al., 2018). Age, as expected, stood out as a fundamental risk factor, matching well-documented trends in heart disease. Meanwhile, resting blood pressure and cholesterol showed more modest influence in some of our plots, which likely reflects the natural variability and measurement limits seen with these indicators in practice.
The moderate level of SHAP interaction values suggests the model behaves mostly in an additive way, rather than depending heavily on complex feature interactions. This actually helps with real-world interpretability, fitting how clinicians usually reason through diagnoses by weighing factors like age, exercise ECG results, and functional capacity as largely independent contributors to risk.
When compared to earlier studies, our results support two consistent themes: ensemble methods tend to outperform others, and functional stress test variables remain central to predicting heart disease (Khurshid et al., 2021; Attia et al., 2019). Random Forest and Gradient Boosting did particularly well here, echoing earlier findings that these methods are robust even with noisy or incomplete clinical data. On the other hand, the neural network’s weaker performance follows what others have noted deep learning models usually need much larger datasets and often don’t work as well with structured, tabular clinical information.
Overall, this work adds to a growing field showing that interpretable machine learning can improve how we predict cardiovascular risk. By bringing together demographic, metabolic, and functional markers into one unified approach, we get a clearer picture of risk. Looking ahead, future research could include longer-term patient data, add multimodal inputs like imaging or time-series ECG readings, and test the model across different hospitals to see how well it generalizes.

# D. CONCLUSION
This research developed and tested a comprehensive machine learning framework to classify heart disease using a dataset of clinical features. By combining thorough exploratory analysis, careful data preparation, and a side-by-side comparison of traditional, ensemble, and deep learning models, we found that the ensemble methods especially Random Forest deliver the best predictive performance for this kind of structured cardiovascular data. The Random Forest model stood out with the highest overall accuracy and ROC–AUC score, while also showing strong sensitivity and a balanced error profile. This makes it a particularly good fit for clinical screening applications. Importantly, our interpretability checks confirmed that the model’s reasoning aligns with medical understanding. The most influential predictors it identified like maximum heart rate during exercise, ST depression, type of chest pain, cholesterol levels, and age are all well-known markers tied to exercise tolerance, metabolic health, and ischemic response. These results not only match established cardiovascular risk factors but also show how interpretable machine learning can support, rather than replace, a clinician’s own judgment. In short, this work demonstrates that advanced ensemble learning techniques can successfully uncover complex, non-linear patterns in clinical data while still being transparent and clinically meaningful. The framework we’ve presented offers a solid, reproducible foundation for assessing heart disease risk and underscores the practical value of machine learning as a supportive tool in cardiovascular care.

# E. LIMITATIONS 
Several limitations should be acknowledged. First, the dataset size remains moderate, which may limit the performance of deep learning models and reduce generalisability to broader populations. Second, the dataset contains missing values particularly for the slope, ca, and thal features which required imputation. Although median/mode imputation is a widely accepted approach, it may introduce bias or reduce variance. Handling missingness differently (e.g., via multiple imputation or model based imputation) may yield further gains. Third, the outcome variable was converted into a binary classification problem. While clinically justified, the original multi-class nature of disease severity was not fully explored in this analysis. Multi-class modelling or ordinal classification may offer deeper insights. Additionally, this study relied on structured clinical variables; unstructured clinical text, imaging, and longitudinal data were not included but may significantly enhance predictive capability. Finally, external validation on an independent cohort was not performed, and future research should evaluate model performance across diverse clinical settings.

# F. FUTURE RESEARCH DIRECTIONS
Although the proposed framework performs well, several promising paths for future research could broaden its impact and usefulness. First, subsequent studies should include larger and more diverse patient groups, ideally gathered from multiple hospitals or healthcare systems. This kind of cross institutional validation would help confirm the model’s real world reliability and minimize any bias tied to a specific population. Second, moving beyond a simple yes/no diagnosis to predict multiple classes or levels of disease severity could offer finer grained insights for clinicians. This would aid in personalizing treatment plans and stratifying risk more precisely. Incorporating longitudinal patient data could also shift the focus from a single snapshot to modelling how heart disease develops over time. Third, future work could look at bringing in different types of data. Combining the structured clinical variables used here with unstructured information like raw ECG waveforms, medical images, or notes from electronic health records might boost predictive accuracy and clinical relevance even further. Finally, before such tools become part of everyday practice, prospective studies are needed to examine real-time clinical use, how clinicians interact with AI recommendations, and critical ethical issues. This includes actively checking for bias, ensuring fairness across patient groups, and building clinician trust. By pursuing these directions, future research can build on our findings to further advance the role of explainable AI in cardiovascular medicine.


# REFERENCE 
1.	Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning.
2.	Beam, A. L., & Kohane, I. S. (2018). Big data and machine learning in health care.
3.	Benjamin, E. J., et al. (2019). Heart disease and stroke statistics—American Heart Association.
4.	Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
5.	D’Agostino, R. B., et al. (2008). The Framingham cardiovascular risk score.
6.	Dorogush, A. V., Ershov, V., & Gulin, A. (2018). CatBoost: Gradient boosting with categorical features.
7.	Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning.
8.	Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree.
9.	Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions (SHAP).
10.	Shahid, N., Rappon, T., & Berta, W. (2020). Applications of machine learning in health care.
11.	World Health Organization. (2021). Cardiovascular diseases fact sheet.
12.	Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning.
13.	Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
14.	D’Agostino, R. B., et al. (2008). General cardiovascular risk profile for use in primary care.
15.	Dorogush, A. V., Ershov, V., & Gulin, A. (2018). CatBoost: Gradient boosting with categorical features.
16.	Khan, M. A., et al. (2020). Deep learning approaches for heart disease prediction.
17.	Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree.
18.	Kumari, V., & Chitra, R. (2013). Classification of heart disease using machine learning methods.
19.	Nahar, J., et al. (2013). Computational intelligence for heart disease diagnosis.
20.	Shahid, N., Rappon, T., & Berta, W. (2020). Applications of machine learning in health care.
21.	Son, Y. J., et al. (2012). Application of data mining to heart disease classification.
22.	Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning.
23.	Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
24.	Dorogush, A. V., Ershov, V., & Gulin, A. (2018). CatBoost: Gradient boosting with categorical features.
25.	Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree.
26.	Kumari, V., & Chitra, R. (2013). Classification of heart disease using machine learning methods.
27.	Lundberg, S. M., & Lee, S.-I. (2017). SHAP: A unified approach to interpreting model predictions.
28.	Shearer, C. (2000). The CRISP–DM model: The new blueprint for data mining.






















