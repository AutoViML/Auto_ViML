import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoviml",
    version="0.1.652",
    author="Ram Seshadri",
    # author_email="author@example.com",
    description="Automatically Build Variant Interpretable ML models fast - now with CatBoost!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/AutoViML/Auto_ViML",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "ipython",
        "jupyter",
        "xgboost==0.90",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn>=0.22",
        "catboost",
        "textblob",
        "nltk",
        "regex",
        "vaderSentiment",
        "imbalanced-learn",
    	"beautifulsoup4",
    	"gensim"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
