# -*- coding: utf-8 -*-
################################################################################
#     Auto_ViML - Automatically Build Multiple Interpretable ML Models in Single Line of Code
#     Python v3.6+
#     Created by Ram Seshadri
#     Licensed under Apache License v2
################################################################################
# Version
from .__version__ import __version__, __nlp_version__
from autoviml.Auto_ViML import Auto_ViML
from autoviml.Auto_NLP import Auto_NLP
if __name__ == "__main__":
    module_type = 'Running'
else:
    module_type = 'Imported'
viml_version_number = __version__
nlp_version_number = __nlp_version__
print("""%s Auto_ViML version: %s. Call using:
             m, feats, trainm, testm = Auto_ViML(train, target, test,
                            sample_submission='',
                            scoring_parameter='', KMeans_Featurizer=False,
                            hyper_param='RS',feature_reduction=True,
                             Boosting_Flag='CatBoost', Binning_Flag=False,
                            Add_Poly=0, Stacking_Flag=False,Imbalanced_Flag=False,
                            GPU_flag=False, verbose=1)
            """ %(module_type, viml_version_number))
print()
###########################################################################################
print("""%s Auto_NLP version: %s.. Call using:
     train_nlp, test_nlp, nlp_pipeline, predictions = Auto_NLP(
                nlp_column, train, test, target, score_type='balanced_accuracy',
                modeltype='Classification',top_num_features=200, verbose=0,
                build_model=True)""" %(module_type, nlp_version_number))
########################################################################
