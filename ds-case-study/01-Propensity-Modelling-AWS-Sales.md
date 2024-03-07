Problem
----------

How would you help the AWS sales team identify key attributes that 
predict whether a customer will repeat purchase on AWS products?

Tips
----------
To ace this case problem, consider the following framework:

1. Ask Clarifying Questions - What does the interviewer mean by “repeat” purchase?
        What AWS product sales need to be optimized?
2. Statistical Modeling - What data do you need for the problem? 
        How would you build a model that predicts repeat purchases?
3. Identifying Key Attributes - How would you identify key attributes that can help the AWS team identify customers that are likely to purchase?
4. Business Impact - How would this prediction model serve the AWS sales team?

Solution
----------
**[Interviewer]** How would you identify key attributes that predict whether a customer will repeat purchase on AWS products?

**[Candidate]** Thank you for the question. I can tell that this problem involves propensity modeling given that the problem involves finding attributes that predict customer conversion. To address this problem, I have a few questions.

**[Interviewer]** Sure thing.

**[Candidate]** I know that the AWS service provides various business solutions from computing, database, and AI tools. Do I need to predict which product a particular customer will likely convert to again? Or, should I focus on whether a customer will convert again at all?  

**[Interviewer]** It will be the latter.

**[Candidate]** Thank you for the clarification. I have another question: what do you mean by a “repeat purchase?” I assume that the AWS sales team provides enterprise solutions. In this case, they most likely care about long-term usages such as annual subscriptions or contracts. Would the repeat purchase be based on if they purchased a product then another one immediately? Or, would it be based on a long-term horizon, meaning that a customer purchased a product this year and again the following year. 

***Commentary***: Notice that the candidate ensures that he clearly understands the scope of the problem before providing a solution. A business problem in data science is often vague. In data science case rounds, demonstrating to the interviewer how you can translate a vague problem into a concrete objective is vital.

**[Interviewer]** Good question. In this case, a repeat purchase occurs when a customer purchased a product the first year and purchased again the following year. 

**[Candidate]** Great, thank you. I will now brainstorm a solution.

**[Interviewer]** Great. Let’s start with data. What data would you use to solve this problem? 

**[Candidate]** I can think of three types of data: (1) user profile (2) user-product behavior, and (3) marketing engagements. Each data type contains the following attributes:

User profile - business name, industry, years in operation, and location. 

User-product behavior - the type of product purchased and engagement metrics (i.e. average # of active sessions the past 1 month), and new products browsed on the AWS platform.

Marketing engagements - newsletter sign-ups, free-trial enrollments and advertisement clicks.

**[Interviewer]** Great. How would you use the data to solve the problem?

**[Candidate]** I would conduct an exploratory data analysis to understand how the attributes correlate with the target variable - the recurrent purchase of a product the following year. In addition, the EDA informs how I should clean and preprocess the data for modeling. 

**[Interviewer]** Okay, how would you create the label for the model? 

**[Candidate]** The label can be derived from the purchase history. I would create an indicator variable with a value of “1” if a purchase occurred again in the following year. Otherwise, the indicator is 0. To create this label set, I would need at least two years of data. 

**[Interviewer]** Okay, what else can you tell me about your model solution?

**[Candidate]** Once I have performed EDA and created labels, I can perform feature engineering and feature selection to identify key signals for a model. Since the target variable is binary, I can use a model with high interpretability such as logistic regression or random forest.

**[Interviewer]** How would you measure the model performance? 

**[Candidate]** Based on whether the label is imbalanced or not, I would use either the accuracy or AUC to measure the model performance. If the ratio between the repeat purchase and no repeat purchase is 1:1, using I would use the accuracy metric. If the ratio is 1:9, using the AUC metric is ideal given that the accuracy metric could be inflated.

**[Interviewer]** Let’s say you have a decent model. Now, how would you address the main question on identifying attributes that lead to a repeat purchase? 

**[Candidate]** If I am using the logistic regression model, I can first filter on predictors that achieve statistical significance, then rank the predictors based on the coefficient values. The ranking helps me identify key predictors that drive repeat purchases. In addition, I can use the partial dependence plot (PDP) to understand the probability of conversion across a variable range while holding other variables constant. This technique can isolate a subset of values in a categorical variable or a bracket within a continuous variable that is highly correlated with the repeat conversion. 

**[Interviewer]** How do you think the AWS sales team could use the key attributes you identified to improve sales?

**[Candidate]** There are two ways this could be helpful for the sales team. First, the propensity model helps detect users that are most likely to convert again. The sales team can invest their efforts in high-intent customers as there are less time and $ cost involved in converting those users compared to the low-intent customers. Secondly, once the high-intent users are identified, I can create customer segments based on key attributes. The segments are helpful for the sales team in creating custom sales funnel for each segment.