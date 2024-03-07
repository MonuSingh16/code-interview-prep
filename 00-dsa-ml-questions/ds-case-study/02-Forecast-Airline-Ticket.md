Problem
----------

The Google interviewer asks:

1. How would you forecast airline ticket prices?
2. How would you correct the Covid impact on the forecast?

Tips
----------
To address this product case question, structure the response in the following manner:

1. Ask clarifying questions - Get clarity on the forecast horizon of the problem.
2. Data - What data can you leverage to build the forecasting model?
3. Methodology - Consider the model design to forecasting the model and the evaluation technique to measure performance.

Solution
----------

[Interviewer] How would you forecast airline ticket prices?

[Candidate] Thank you for the problem. I suppose Google’s search has an airline price prediction that allows users to pick a day to travel based on the price predicted. This is a helpful feature that provides a lot of value to users. Before I start, I have a couple of questions to frame the problem. Do you mind if I ask you? 

[Interviewer] Sure thing. 

[Candidate] Do I need to forecast the ticket prices on a short-term horizon or long-term horizon? A short-term horizon could be daily prices over the course of one month. A long-term horizon could be daily prices over the course of three or more months.

[Interviewer] That’s a good question. We want to anticipate the daily price six months ahead.

[Candidate] Thank you for the clarification. I have another question. What kind of dataset do I have to build the forecasting model?

[Interviewer] Let’s say you have three years of historical price data and any macroeconomic signals of your choice.   

[Candidate] Great thank you. To start, I think it makes the most sense to build a univariate model to get a benchmark, then incorporate macroeconomic signals to improve the forecasting accuracy. Can I start with this process?

[Interviewer] Yes, sounds good. 

[Candidate] To start, I need to conduct an exploratory data analysis to apply an appropriate modeling procedure. I would conduct the following analysis that checks for: (1) missingness, (2) outliers, (3) seasonality, (4) trends, and (5) ARMA and MA processes.

[Interviewer] Okay, what’s the next step?

[Candidate] Given that the problem is long-term forecasting, I won’t be able to use LSTM or univariate models such as SARIMA. Such models are designed for short-term models. 

Commentary: The candidate is partially correct. There are advanced TS modeling techniques that enable long-term forecasting using short-term models. One such technique is the rolling method.

For instance, suppose that the SARIMA model forecasts 7 days, and the objective is to forecast the next 28 days. Take the output from the first 7 days of the forecast as the input for the next 7 days (days 8 to 14). Continue this procedure until you have a forecast for the entire 28 days. 

[Interviewer] Okay, what’s the alternative then?

[Candidate] I could start with a linear model which enables extrapolation. I presume that airline prices are seasonal. For instance, during the summer (in the Northern hemisphere), airline prices would rise as the demand for travel increases. I can build a linear model that regresses time-based signals, including hour, day of the week, week of the month, month, season, and year. 

[Interviewer] Okay, how would you build separate predictions depending on locations and airlines? 

[Candidate] I could consider two designs. Ultimately, the model performance will determine which one to use. The first design is to build a separate model per location and airline combination. The second design is to build a single model with locations and airlines encoded as categorical features. 

[Interviewer] Okay, how would you measure the model performance?

[Candidate] I can evaluate the time-series model using the mean absolute percentage error (MAPE). This metric allows interpretability of the error that’s easy for stakeholders to understand. 

[Interviewer] Sounds good. Now, suppose you have macroeconomic signals. What kind of signals would you use? 

[Candidate] Can I presume I have signals such as unemployment rate, GDP growth, and population density, and such? 

[Interviewer] Yup. Can you elaborate on how those could be helpful?

[Candidate] Well, prices fluctuate depending on the demand. Demand fluctuates depending on the economic conditions. If the economy is bad, people spend less, demand drops, and, ultimately, the price drops. 

[Interviewer] Now one final question. How would you accommodate for the Covid impact on the time-series forecast?

[Candidate] That’s an interesting problem. I think leveraging external signals such as lockdown indicators and the Covid confirmed case rates could help build a model that accommodates the Covid impact.

