# Auto_ViML
Automated Variant Interpretable Machine Learning project. Created by Ram Seshadri. Permission Granted Upon Request.

    ####   Automatically Build Variant Interpretable Machine Learning Models (Auto_ViML) ######
    # Auto_ViML was designed for building a High Performance Interpretable Model With Fewest Vars. 
    ##  The V is for Variant because it tries Multiple Models and Multiple Features to find the
    #   Best Performing Interpretable Model with the Least Amount of Features for any data set.
    #####   It is built mostly using Scikit-Learn, Numpy, Pandas and Matplotlib. The only
    ####  importation you may have to do is "shap" but if you don't have it, it will skip it.
    ####   Just send in train as a file or a dataframe, similarly a test, a submission file 
    ###    (if any) and it will build the model. You have a number of Flags to tune the result.
    It will return a trained model and your submission file will be "mysubmission.csv" file. 
    So you can take it and submit it to competitions or hackathons right away.
    If no submission file was given but as long as you give a test file, it will create
    a submission file for you named "mySubmission.csv". It returns XGB and important_features.
    It also returns the modified Train and Test files. Great to do more modeling if necessary.

# Steps:
1. Copy or download this Auto_ViML.py file to any directory. 
1. In the same directory, open a Jupyter Notebook and use this line to import the .py file:

    from Auto_ViML import Auto_ViML

1. Then use Pandas to import a CSV or other data file into a dataframe and split it into Train and Test data frames. Notice that Auto_ViML uses DataFrames as inputs. Do not send in Numpy Arrays since it will give an error.

1. Finally, call Auto_ViML using the train, test dataframes and the name of the target variable in data frame. That's all.

Auto_ViML(train, target, test='',sample_submission='',modeltype='Classification',
            scoring_parameter='logloss', Boosting_Flag=None,
            Add_Poly=0, Stacking_Flag=False, Binning_Flag=False,
              Imbalanced_Flag=False, verbose=0)
              
Hope this helps. You don't need any special Libraries other than whatever is in your Anaconda Python Distribution. 

Also, BTW, Auto_ViML runs on Python 3 versions without problems. But I suspect it can run on Python 2 as well.
