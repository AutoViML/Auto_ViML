# -*- coding: utf-8 -*-
################################################################################
#     Auto_ViML - Automatically Build Multiple Interpretable ML Models in Single Line of Code
#     Python v3.6+
#     Created by Ram Seshadri
#     Licensed under Apache License v2
################################################################################
# Version
from .__version__ import __version__
if __name__ == "__main__":
    module_type = 'Running'
else:
    module_type = 'Imported'
version_number = __version__
print("""Imported Auto_ViML version: %s. Call using:
             m, feats, trainm, testm = Auto_ViML(train, target, test,
                            sample_submission='',
                            scoring_parameter='', KMeans_Featurizer=False,
                            hyper_param='RS',feature_reduction=True,
                             Boosting_Flag='CatBoost', Binning_Flag=False,
                            Add_Poly=0, Stacking_Flag=False,Imbalanced_Flag=False,
                            verbose=1)
            """ %version_number)
print('Now Auto_ViML can solve multi-label, multi-output problems. Also Auto_NLP included.')
print('To get the latest version, perform "pip install autoviml --no-cache-dir --ignore-installed"')
###########################################################################################
