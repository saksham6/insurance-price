# insurance-price-prediction
## Table of Contents  
1. [Prerequisites](#prerequisites)
1. [Problem-Statement](#problem-statement) 

1. [Data-Analysis-And-Data-Cleaning(EDA](#data-analysis-and-data-cleaning)

1. [Model Selection](#model-selection)

1. [Deployment](#deployment)
1. [Visit Webpage for prediction](https://insurance-price-predictions.herokuapp.com/)


 

<!-- <a name="header"/> -->
## Prerequisites
1. Scikit Learn
1. Pandas
1. Numpy
1. Flask

## Problem-Statement  

The objective of this exercise is to build a model, using data that provide the 
optimum insurance cost for an individual. i have used the health and habit related parameters for 
the estimation cost of insurance

## Data-Analysis-And-Data-Cleaning

There are few columns where there we can see bit skewness but overall we can data is normally distributed.

1. **Missing Value**
    * We have missing value in Year_last_admitted and BMI variable if we look into missing value count which is 11881 and 990 respectively.  Now if we calculate missing value for Year_last_admitted around 50% of data are missing now to treat such large number of data will create nothing but bias and we don’t want to create any bias which building a model. So, it is wise to remove all missing data for Year_last_admitted or entire column depending upon it significant but if we look into BMI missing value which is 990 which is around 3 percent of data and as industries standard at maximum, we can treat missing value around 1-5% of data. So, we can treat missing value of BMI with median.  
      
      Meanwhile if we looked into Smoking_Status variable we can notice we have no missing data but we have some data as Unknown. Now this data either can be missing or customer not opt to say but we can’t treat it as valid data. In such case either we need to treat or drop such data. If we count for unknown data its around 30% of data (7555) which is again huge data and treating such data will create bias.  

1. **Outlier**
   * we have outlier in daily_avergae_steps and BMI variable. since there is just 1-2 people falling into BMI categories, we can keep the data whereas for daily_avergae_steps it is quite possible that people who are more active or do extreme exercise can walk more than 10000 steps so we don’t want to lose such valid information by treating them. Hence, we are not treating outliers. It seems all are valid.  
  
  1. **Removal of unwanted variables (if applicable)**
    
     There are some variables which does not contribute any impact to the model building such as 

     * Applicant ID: it only gives us the personal id number which does not have anything to do with model building.

     * Location: Now this is an optional,we can see number wise, we have Bangalore has highest number but percentage wise if we look then all cities have around same number of hospital where people admitted which means our model learn nothing different.
    Now once model is built it never remain same for future as well as we need   to build model in certain interval as data evolves. Data keeps on changing so we need to build model accordingly. So, if the data shows that there is fluctuation in location then building a model with such variable would make much sense. In short, I would say it can be use and remove on the basis of data availability.

     * Details of Year_last_admitted: This variable has almost had 11881 
missing data around 50% and we can’t treat 50% of data since it will introduce bias in our dataset. Now dropping NaN value will lose 50% of data.
But if we look how important is this variable, we can either do some test or we can analysis on the basis looking other data. 

       If we look this variable this will tell us when person was admitted last time which means they have some disease before and we have already such variable in our dataset called other_major_decs_history and heart_decs_history

       So, to keep similar variable which will cause 50% data loss. Its better to drop entire column and we can save other data.

       However model building is iterative process we have to tune parameter, transform of data etc so we can also try with and without dropping variable and see if it makes any difference to model building.


## Model-Selection
| Model Name                | Accuracy Train| Accuracy Test|
| -------------             |:------------- |--------------|
| Linear Regression         | 94.1%         | 93.7%        |
| Random Forrest Regressor  | 94.7%         | 92.5%        |
| Xgboost Regressor         | 97.27%        | 95.24%       |
| Gradient Boosting         | 95.9%         | 93.4%        |
| Lasso                     | 94.1%         | 93.8%        |
| Ridge                     | 94.1%         | 93.7%        |



## Deployment
  * ### **Deploy into Heroku using Flask**
     
    Flask is a web framework that provides libraries to build
lightweight web applications in python.
  
    It is based on WSGI toolkit and jinja2 template engine.
WSGI is an acronym for web server gateway interface which is a standard for
python web application development.
  
    Jinja2 is a web template engine to render the dynamic web pages.
  
    Flask requires python 2.7 or higher  
      
   * ### **Steps to deploy model into Heroku using Flask**
     
     * **Step-1**
       
       Create a “pickle” of the model (i.e. file with an extension as .pkl)
       “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream
  
            **save the model to disk**
  
         Import pickle
         pickle_out = open("adult_flask.pkl","wb")  

            pickle.dump(model, pickle_out)  

            loaded_model = pickle.load(open("adult_flask.pkl", "rb"))  

            result = loaded_model.score(X_test, y_test)
  
            print(result)  


      * **Step-2**
          
          Create an HTML page/file with or without CSS which would
be the index page
One Page could be created both to receive the user input and display the result or
separate page could be created to receive input and then to display result.
These files should have an extension as “.html” and these files should be stored
in the folder “templates”

     * **Step-3**  

       Create an file with an extension “.py” using python editor like
jupyter notebook.

        Import the library files
import numpy as np
import pandas as pd
from flask import Flask, request,render_template
import pickle  

        Initialise the flask
        app= Flask(__name__)  
        Refer app.py file

     * **Step-4**  
        * Folder structure
        1. Flask
        1. Templates  

            1. Home.html
            1. result.html
        1. adult_flask.pkl
        1. app.py  

      
      * **Step-5**
        
        @ Anconda prompt change the directory to where you have all the files like in our case,
    folder is “Flask”. Once you are at the required directory type   
      
            python app.py   
  

      
      * **Steps-6**
  
        to deploy on heroku we need three files.
              
              1. Procfile
              2. runtime.txt
              3. requirements.txt 
    
        **Procfiles:**
      Note: its should have any extension and "P" in Procfile should in uppercase.
        
        **runtime.txt:**
      we need to declare python version.

        **requirements.txt**
    all the dependecise libraries used in you model should be written here.


      
      * **Step-7**
  
        Login into heroku.
          
          create new app
            
            Go to deploy tab
            there will be two option using Heroku CLI and Github. select any of them and all the steps will be shown below. 
