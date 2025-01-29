import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoviml",
    version="0.2.0",
    author="Ram Seshadri",
    # author_email="author@example.com",
    description="Automatically Build Variant Interpretable ML models fast - now with CatBoost!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/AutoViML/Auto_ViML",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "numpy<2",
        "pandas>=2.0",
        "scipy<=1.11.4",
        "xlrd",
        "matplotlib>3.7.4",
        "beautifulsoup4",
        "emoji",
        "ipython",
        "jupyter",
        "seaborn",
        "catboost>=1.2.7",
        "textblob",
        "nltk",
        "regex",
        "xgboost>=0.82,<=1.7.6",
        "vaderSentiment",
        "imbalanced-learn>=0.10.1",
        "shap>=0.36.0",
        "scikit-learn>=0.24,<=1.5.2",
        "lightgbm>=3.0.0",
        "networx",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
