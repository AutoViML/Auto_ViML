<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><link rel="stylesheet" type="text/css" href="/+static/base.vsKBklzePi_Td7VvkjGVKw.cache.css"/><link rel="stylesheet" type="text/css" href="/+static/doc.DKuU4ej5LMTa95NGDTR2Pw.cache.css"/><link rel="stylesheet" type="text/css" href="/+static/prettify/prettify.pZ5FqzM6cPxAflH0va2Ucw.cache.css"/><!-- default customHeadTagPart --></head><body class="Site"><header class="Site-header "><div class="Header"><div class="Header-title"></div></div></header><div class="Site-content Site-Content--markdown"><div class="Container"><div class="doc"><h1><a class="h" name="Auto_ViML" href="#Auto_ViML"><span></span></a><a class="h" name="auto_viml" href="#auto_viml"><span></span></a>Auto_ViML</h1><h2>Automatically Build Variant Interpretable ML models fast!</h2>
    #########################################################################################################
    #############       This is not an Officially Supported Google Product!         #########################
    #########################################################################################################
    #Copyright 2019 Google LLC                                                                        #######
    #                                                                                                 #######
    #Licensed under the Apache License, Version 2.0 (the "License");                                  #######
    #you may not use this file except in compliance with the License.                                 #######
    #You may obtain a copy of the License at                                                          #######
    #                                                                                                 #######
    #    https://www.apache.org/licenses/LICENSE-2.0                                                  #######
    #                                                                                                 #######
    #Unless required by applicable law or agreed to in writing, software                              #######
    #distributed under the License is distributed on an "AS IS" BASIS,                                #######
    #WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         #######
    #See the License for the specific language governing permissions and                              #######
    #limitations under the License.                                                                   #######
    #########################################################################################################
    ####       Automatically Build Variant Interpretable Machine Learning Models (Auto_ViML)           ######
    ####                                Developed by Ramadurai Seshadri                                ######
    ######                               Version 0.56                                               #########
    #########################################################################################################
    ####   Auto_ViML was designed for building a High Performance Interpretable Model With Fewest Vars.   ###
    ####   The "V" in Auto_ViML stands for Variant because it tries Multiple Models and Multiple Features ###
    ####   to find the Best Performing Model for any data set.The "i" in Auto_ViML stands " Interpretable"###
    ####   since it selects the fewest Features to build a simpler, more interpretable model. This is key. ##
    ####   Auto_ViML is built mostly using Scikit-Learn, Numpy, Pandas and Matplotlib. Hence it should run ##
    ####   on any Python 2 or Python 3 Anaconda installations. You won't have to import any special      ####
    ####   Libraries other than "SHAP" library for SHAP values which provides more interpretability.    #####
    ####   But if you don't have it, Auto_ViML will skip it and show you the regular feature importances. ###
    #########################################################################################################
    ####   INPUTS:                                                                                        ###
    #########################################################################################################
    ####   train: could be a datapath+filename or a dataframe. It will detect which is which and load it.####
    ####   test: could be a datapath+filename or a dataframe. If you don't have any, just leave it as "". ###  
    ####   submission: must be a datapath+filename. If you don't have any, just leave it as empty string.#### 
    ####   target: name of the target variable in the data set.                                          ####
    ####   sep: if you have a spearator in the file such as "," or "\t" mention it here. Default is ",". ####
    ####   scoring_parameter: if you want your own scoring parameter such as "f1" give it here. If not, #####
    ####       it will assume the appropriate scoring param for the problem and it will build the model.#####
    ####   hyper_param: Tuning options are GridSearch ('GS') and RandomizedSearch ('RS'). Default is 'GS'.###
    ####   Boosting Flag: you have 3 possible choices (default is False):                               #####
    ####    None = This will build a Linear Model                                                       #####
    ####    False = This will build a Random Forest or Extra Trees model (also known as Bagging)        #####
    ####    True = This will build an XGBoost model                                                     #####
    ####   Add_Poly: Default is 0. It has 2 additional settings:                                        #####
    ####    1 = Add interaction variables only such as x1*x2, x2*x3,...x9*10 etc.                       ##### 
    ####    2 = Add Interactions and Squared variables such as x1**2, x2**2, etc.                       #####
    ####   Stacking_Flag: Default is False. If set to True, it will add an additional feature which     #####
    ####         is derived from predictions of another model. This is used in some cases but may result#####
    ####         in overfitting. So be careful turning this flag "on".                                  #####
    ####   Binning_Flag: Default is False. It set to True, it will convert the top numeric variables    #####
    ####         into binned variables through a technique known as "Entropy" binning. This is very     #####
    ####         helpful for certain datasets (especially hard to build models).                        #####
    ####   Imbalanced_Flag: Default is False. If set to True, it will downsample the "Majority Class"   #####
    ####         in an imbalanced dataset and make the "Rare" class at least 5% of the data set. This   #####
    ####         the ideal threshold in my mind to make a model learn. Do it for Highly Imbalanced data.#####
    ####   verbose: This has 3 possible states:                                                         #####
    ####    0 = limited output. Great for running this silently and getting fast results.               #####
    ####    1 = more charts. Great for knowing how results were and making changes to flags in input.   #####
    ####    2 = lots of charts and output. Great for reproducing what Auto_ViML does on your own.       #####
    #########################################################################################################
    ####   OUTPUTS:                                                                                     #####
    #########################################################################################################
    ####   model: It will return your trained model                                                     #####
    ####   features: the fewest number of features in your model to make it perform well                #####
    ####   train_modified: this is the modified train dataframe after removing and adding features      #####
    ####   test_modified: this is the modified test dataframe with the same transformations as train    #####
    #################               A D D I T I O N A L    N O T E S                              ###########
    ####   Finally, it writes your submission file to disk in the current directory called "mysubmission.csv"
    ####   This submission file is ready for you to show it clients or submit it to competitions.       #####     
    ####   If no submission file was given but as long as you give it a test file name, it will create  #####
    ####   a submission file for you named "mySubmission.csv".                                          #####
    ####   Auto_ViML works on any Multi-Class, Multi-Label Data Set. So you can have many target labels #####  
    ####   You don't have to tell Auto_ViML whether it is a Regression or Classification problem.       #####
    ####   Suggestions for a Scoring Metric:                                                            #####
    ####   If you have Binary Class and Multi-Class in a Single Label, Choose Accuracy. It will        ######
    ####   do very well. If you want something better, try roc_auc even for Multi-Class which works.   ######
    ####   You can try F1 or Weighted F1 if you want something complex or for Multi-Class.             ######
    ####   Note that For Imbalanced Classes (<=5% classes), it automatically adds Class Weights.       ######
    ####   Also, Note that it handles Multi-Label automatically so you can send Train data             ######
    ####   with multiple Labels (Targets) and it will automatically predict for each Label.            ######
    ####   Finally this is Meant to Be a Fast Algorithm, so use it for just quick POCs                 ######
    ####   This is Not Meant for Production Problems. It produces great models but it is not Perfect!  ######
    ######################### HELP OTHERS! PLEASE CONTRIBUTE! OPEN A PULL REQUEST! ##########################
    #########################################################################################################
 <h1><a class="h" name="DOWNLOAD-INSTALLATION" href="#DOWNLOAD-INSTALLATION"><span></span></a><a class="h" name="download-installation" href="#download-installation"><span></span></a>DOWNLOAD / INSTALLATION</h1><ol><li>Copy or download this entire directory of files to any local directory using git clone or any download methods.</li></ol><h1><a class="h" name="RUN-AUTOViML" href="#RUN-AUTOViML"><span></span></a><a class="h" name="run-autoviml" href="#run-autoviml"><span></span></a>RUN AUTO_ViML</h1><ol start="2"><li><p>In the same directory, open a Jupyter Notebook and use this line to import the .py file: <br>from Auto_ViML import Auto_ViML</p></li><li><p>Load a data set (any CSV or text file) into a Pandas dataframe and split it into Train and Test dataframes. If you don't have a test dataframe, you can simple assign the test variable below to '' (empty string):</p></li></ol><p>Finally, call Auto_ViML using the train, test dataframes and the name of the target variable in data frame. That's all.<br>Auto_ViML(train, target, test,sample_submission=’’,modeltype=‘Classification’, scoring_parameter=‘logloss’, Boosting_Flag=None, Add_Poly=0, Stacking_Flag=False, Binning_Flag=False, Imbalanced_Flag=False, verbose=0)</p><h1><a class="h" name="DISCLAIMER" href="#DISCLAIMER"><span></span></a><a class="h" name="disclaimer" href="#disclaimer"><span></span></a>DISCLAIMER</h1><p>“This is not an official Google product”.</p><h1><a class="h" name="LICENSE" href="#LICENSE"><span></span></a><a class="h" name="license" href="#license"><span></span></a>LICENSE</h1><p>Licensed under the Apache License, Version 2.0 (the &ldquo;License&rdquo;).</p></div></div></div><!-- default customFooter --><footer class="Site-footer"><div class="Footer"><span class="Footer-poweredBy">Powered by <a href="https://gerrit.googlesource.com/gitiles/">Gitiles</a>| <a href="https://policies.google.com/privacy">Privacy</a></span><div class="Footer-links"></div></div></footer></body></html>
