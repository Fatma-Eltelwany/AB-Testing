#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df = pd.read_csv("ab_data.csv")
df.head()


# In[3]:


df.info()


# b. Use the below cell to find the number of rows in the dataset.

# In[4]:


df.shape[0]


# c. The number of unique users in the dataset.

# In[5]:


len(df.user_id.unique())


# d. The proportion of users converted.

# In[6]:


df[df.converted == 1].shape[0]/294478


# e. The number of times the `new_page` and `treatment` don't line up.

# In[7]:


df_treatmentonly = df.query('group == "treatment" & landing_page != "new_page"')
df_newpageonly = df.query('group != "treatment" & landing_page == "new_page"')
df_treatmentonly.shape[0] + df_newpageonly.shape[0]


# In[8]:


df_treatmentonly.sample(10)


# f. Do any of the rows have missing values?

# In[9]:


df.isna().sum().sum()


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[10]:


df2 = pd.concat([df,df_treatmentonly,df_newpageonly]).drop_duplicates(keep = False)
df2.info()


# In[11]:


df.info()


# In[12]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[13]:


len(df2.user_id.unique())


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[14]:


df2.user_id.value_counts()


# c. What is the row information for the repeat **user_id**? 

# In[15]:


df2.loc[df2['user_id']==773192]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[16]:


df2.drop_duplicates(subset= "user_id",inplace = True)


# In[17]:


df2.loc[df2['user_id']==773192]


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[18]:


df2.converted.value_counts()[1]/290584


# 
# 
# 
# 
# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[19]:


df_control = df2.query('group == "control"')
p_control_converted = df_control.query('converted == 1').shape[0]/df_control.shape[0]
p_control_converted


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[20]:


df_treatment = df2.query('group == "treatment"')
p_treatment_converted = df_treatment.query('converted == 1').shape[0]/df_treatment.shape[0]
p_treatment_converted


# d. What is the probability that an individual received the new page?

# In[21]:


p_newpage = df2.query('landing_page == "new_page"').shape[0]/df2.shape[0]
p_newpage


# e. Consider your results from a. through d. above, and explain below whether you think there is sufficient evidence to say that the new treatment page leads to more conversions.

# In[22]:


#computes the difference between the probablilties of converting in the treatment and control groups
diff = p_treatment_converted - p_control_converted
print(diff)


# **Answer** \
# The difference between the two probabilities is so small. Thus, there is not enough evidence in favor of the new page.

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **Answer** \
# $H_{0}: p_{new} - p_{old} =< 0$ 
# 
# 
# $H_{1}: p_{new} - p_{old} > 0$

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[23]:


p_new = df2.converted.mean()
p_new


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[24]:


#Under the null, the two converted rates are equal
p_old = p_new
p_old


# c. What is $n_{new}$?

# In[25]:


n_new = df2.query('landing_page == "new_page"').shape[0] 
#or
#df2.query('group == "treatment"').shape[0]
n_new


# d. What is $n_{old}$?

# In[26]:


n_old = df2.query('landing_page == "old_page"').shape[0]
#or
#df2.query('group == "control"').shape[0]
n_old


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[27]:


new_page_converted = np.random.choice([1,0], size = n_new, p=[p_new, (1-p_new)])
new_page_converted


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[28]:


old_page_converted = np.random.choice([1,0], size = n_old, p=[p_old, 1-p_old])
old_page_converted


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[29]:


#p_new - p_old from simulation
new_page_converted.mean() - old_page_converted.mean()


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in a numpy array called **p_diffs**.

# In[30]:


#bootstrapping
p_diffs = []
for _ in range(10000):
    boot_samp = df2.sample(df2.shape[0], replace = True)
    new_df = boot_samp.query('landing_page == "new_page"')
    old_df = boot_samp.query('landing_page == "old_page"')
    boot_pnew = new_df.query('converted == 1').shape[0]/new_df.shape[0]
    boot_pold = old_df.query('converted == 1').shape[0]/old_df.shape[0]
    p_diffs.append(boot_pnew - boot_pold)


# In[31]:


p_diffs = np.array(p_diffs)
p_diffs.mean(), diff


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[32]:


plt.hist(p_diffs);


# In[33]:


null_vals = np.random.normal(0, p_diffs.std(), p_diffs.size )


# In[34]:


plt.hist(null_vals);
plt.axvline(diff, color = 'r');


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[35]:


p_diffs = np.array(p_diffs)
(null_vals > diff).mean()


# k. In words, explain what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **Answer** \
# This is called the p-value. It is the probability of observing our findings given that the null hypothesis is true. The p-value we have here indicates that we fail to reject the null hypothesis. The less the p-value is, the more significant it becomes (that is we reject the null and a decision can be made to adopt the alternative hypothesis).

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[36]:


import statsmodels.api as sm

convert_old = df2.query('group == "control" & converted == 1').shape[0]
convert_new = df2.query('group == "treatment" & converted == 1').shape[0]
n_old = df2.query('group == "control"').shape[0]
n_new = df2.query('group == "treatment"').shape[0]


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[37]:


z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller', prop_var=False)

z_score, p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **Answer** \
# The z-score represents how many standard deviations the observed statistic is from the mean. The p-value is 0.905 which is larger than 0.05 and we reach the same conclusion as before; we fail to reject the null hypothesis.
# 

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Answer** \
# Logistic Regression

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[38]:


df2[['new_page','old_page']]= pd.get_dummies(df2['landing_page'])
df2.drop('old_page', axis = 1, inplace = True)
df2.head()


# In[39]:


df2 = df2.rename(columns={'new_page': 'ab_page'})
df2['intercept'] = 1
df2.head()


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[40]:


lm_1 = sm.Logit(df2['converted'],df2[['intercept','ab_page']])
results_1 = lm_1.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[41]:


results_1.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# **Answer** \
# The p-value associated with ab_page is 0.190 which is still bigger than 0.05 and we still fail to reject the null. In Part II it is 0.9032. The difference can be explained by the type of test each part is performing. Part II performed a one-tailed test where the null is that the conversion rate is less or equal and the alternative is more than. While in part III, the test is two-tailed using equal and not equal signs where the null is that conversion rates for both landing pages are equal and the alternative is that they are unequal.

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Answer** \
# Adding other factors will provide more accurate results and give more insights into how several factors affect the conversion rate. However, if it happens that these factors are not independent of each other, this might give false correlations. That means adding extra factors requires careful analysis to reach a correct conclusion.

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[42]:


countries_df = pd.read_csv('./countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')


# In[43]:


df_new.head()


# In[44]:


df_new.country.value_counts()


# In[45]:


### Creating the necessary dummy variables
df_new[['CA', 'UK', 'US']] = pd.get_dummies(df_new['country'])
df_new.head()


# In[46]:


df_new.drop('CA', axis = 1, inplace = True)
df_new.head()


# In[53]:


#inspecting the average conversion rate for each country for the the two landing pages
df_new.groupby(['country','ab_page']).mean()


# In[54]:


### Fit Your Linear Model And Obtain the Results
lm_2 = sm.Logit(df_new['converted'], df_new[['intercept','US','UK']])
results_2 = lm_2.fit()


# In[55]:


results_2.summary()


# In[56]:


1/np.exp(0.0408), np.exp(0.0507)


# Using Canada as our baseline and the above calculation, we can get that: 
# * Website users located in the US are 0.96 times more likely to convert compared to those located in Canada.
# * Website users located in the UK are 1.05 times more likely to convert compared to those locates in Canada.

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[58]:


### Fit Your Linear Model And Obtain the Results
df_new['UK_ab_page'] = df_new['UK'] * df_new['ab_page']
df_new['US_ab_page'] = df_new['US'] * df_new['ab_page']
lm_3 = sm.Logit(df_new['converted'], df_new[['intercept','ab_page','US','UK','UK_ab_page', 'US_ab_page']])
results_3 = lm_3.fit()


# In[59]:


results_3.summary()


# Both the p-values for UK_ab_page and US_ab_page are greater than 0.05 (we fail to reject the null, that is the country is not relvant to the conversion rate) so it seems the country has almost no effect on the conversion rate. 

# <a id='conclusions'></a>
# ## Conclusions
# 
#  In this projected, we investigated whether a website should change its landing page or not through three approaches: Probablities, A/B Testing, and Logistic Regression. 
#  * From pure probablities, we reached the conclusion that further analysis was needed as not enough evidence was supporting the new page is actually doing any better than the old one.
#  * From A/B testing, we came to the conclusion that we can not reject the null hypothesis as the p-value was not significant (p-value = 0.9032, but might change slightly if run again due to randomness in sampling). We used the z-score as well.
#  * From the regression model, the p-value is different from the one we got in A/B testing (p-value = 0.190). But still it was not significant and we fail to reject the null. Furthermore, we explored the effect, or lack thereof, of where the website users are located might have on the conversion rate. It turned out the counrty does not have much of an influence on the conversion rate.
#  
# Since the experiment kept running for 4 months and the data collected is in the neighborhood of 300,000 unique user, there is no evidence the new page is any better than the old one. The final takeaway is to continue using the old page or to develop a new one and run the experiment again.
#  
