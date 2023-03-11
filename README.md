<h1 align="center">Introduction to Statistical Learning</h1>
<h6 align="center"><small>STAT 387 | Final Project | Winter 2023</small></h6>
<p align="center"><b>#K-nearest neighbor algorithm  &emsp; #Logistic regression &emsp; <br> #Linear Discriminant Analysis &emsp; #Quadratic Discriminant Analysis</b></p>

<p align="center">
<a href="https://github.com/theRealLeif/STAT387" target="_blank">
<img src="README.asset\Logo.svg" width="200"/>
</a>
</p>

<h2 align="center">Preamble</h2>

Consider the wine quality dataset from [UCI Machine Learning Respository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) [^1]. We will focus only on the data concerning white wines (and not red wines). Dichotomize the `quality` variable as `good`, which takes the value 1 if `quality` ≥ 7 and the value 0, otherwise. We will take `good` as response and all the 11 physiochemical characteristics of the wines in the data as predictors.

<h2 align="center">Problem</h2>

Which of the following classifiers would you recommend? Justify your answer.
- K-Nearest Neighbot
- Logistic Regression
- Linear Discriminant Analysis
- Quadratic discriminant analysis

<h2 align="center">File Tree</h2>

```
📦STAT 387 Final Project
 ┣ 📂README.asset                   // Raw Assets for README 
 ┣ 📂lib                            // Supplementary Materials
 ┃ ┣ 📂img
 ┃ ┣ 📄paper.pdf
 ┃ ┣ 📄report.pdf
 ┃ ┗ 📄report.rmd
 ┣ 📂src                            // Source Code
 ┃ ┣ 📂data                         // Raw Data
 ┃ ┃ ┣ 📄winequality-white.csv
 ┃ ┃ ┗ 📄winequality.names
 ┃ ┣ 📄KNN.R
 ┃ ┣ 📄Logit.R
 ┃ ┣ 📄LDA.R
 ┃ ┣ 📄QDA.R
 ┃ ┗ 📄Complete_Code.R
 ┣ 📂usr                            // Member Contributions
 ┃ ┣ 📂Alana
 ┃ ┣ 📂Leif
 ┃ ┗ 📂Sang
 ┃ ┗ 📂web   
 ┣ 📂web                            // Repository Website
 ┣ 📄.gitignore
 ┗ 📄README.md
```

<p align="right">
<a href="https://github.com/theRealLeif/STAT387" target="_blank">
<img src="https://img.shields.io/github/last-commit/theRealLeif/STAT387?label=Last%20commit"/>
</a>
</p>

<h2 align="center">Reference</h2>

[^1]: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
