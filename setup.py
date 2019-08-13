import setuptools

with open("README.md", "r", encoding="utf-16") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoviml",
    version="0.0.2",
    author="Ram Seshadri",
    # author_email="author@example.com",
    description="Automatically Build Variant Interpretable ML models fast!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/AutoViML/Auto_ViML",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "ipython",
        "jupyter",
        "xgboost",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

