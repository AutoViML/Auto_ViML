import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoviml",
    version="0.1.721",
    author="Ram Seshadri",
    # author_email="author@example.com",
    description="Automatically Build Variant Interpretable ML models fast - now with CatBoost!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/AutoViML/Auto_ViML",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "numpy<=1.19.5",
        "pandas>=1.1.3, <2.0",
        "xlrd",
        "matplotlib",
        "beautifulsoup4",
        "emoji",
        "ipython",
        "jupyter",
        "seaborn",
        "catboost",
        "textblob",
        "nltk",
        "regex",
        "xgboost<=1.6.2",
        "vaderSentiment",
        "imbalanced-learn>=0.10.1",
        "shap>=0.36.0",
        "imbalanced_ensemble>=0.2.0",
        "scikit-learn>=0.24,<=1.2.2",
        "lightgbm>=3.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
