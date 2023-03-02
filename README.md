<h1 align="center">Introduction to Statistical Learning</h1>
<p align="center"><b>#K-nearest neighbor algorithm  &emsp; #Logistic regression &emsp; <br> #Linear Discriminant Analysis &emsp; #Quadratic Discriminant Analysis</b></p>

<p align="center">
<a href="https://github.com/theRealLeif/STAT387" target="_blank">
<img src="README.asset\Logo.svg" width="200"/>
</a>
</p>

<h2 align="center">Preamble</h2>

Consider the wine quality dataset from [UCI Machine Learning Respository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) [1]. We will focus only on the data concerning white wines (and not red wines). Dichotomize the `quality` variable as `good`, which takes the value 1 if quality ≥ 7 and the value 0, otherwise. We will take `good` as response and all the 11 physiochemical characteristics of the wines in the data as predictors.

<h2 align="center">Problem</h2>

Which of the following classifiers would you recommend? Justify your answer.
- K-Nearest Neighbot
- Logistic Regression
- Linear Discriminant Analysis
- Quadratic discriminant analysis

<h2 align="center">File Tree</h2>

```
📦STAT 387 Final Project
 ┣ 📂lib                            // Supplementary Materials
 ┃ ┗ 📄paper.pdf
 ┣ 📂README.asset                   // Raw Assets for README 
 ┣ 📂src                            // Source Code
 ┃ ┣ 📂Alana
 ┃ ┣ 📂Leif
 ┃ ┃ ┗ 📄Leif.rmd
 ┃ ┣ 📂Public                       // Collective Work
 ┃ ┃ ┣ 📂data                       // Raw data
 ┃ ┃ ┃ ┣ 📄winequality-white.csv
 ┃ ┃ ┃ ┗ 📄winequality.names
 ┃ ┃ ┣ 📂report                     // Paper Report
 ┃ ┃ ┃ ┣ 📄report_PDF.pdf
 ┃ ┃ ┃ ┗ 📄report_PDF.rmd
 ┃ ┃ ┣ 📄Step.1_KNN.R
 ┃ ┃ ┣ 📄Step.2_Logit.R
 ┃ ┃ ┣ 📄Step.3_LDA.R
 ┃ ┃ ┣ 📄Step.4_QDA.R
 ┃ ┃ ┣ 📄Step.99_Complete_Code.html
 ┃ ┃ ┣ 📄Step.99_Complete_Code.R
 ┃ ┃ ┗ 📄Step.99_Complete_Code.rmd
 ┃ ┃ ┣ 📄Public.html
 ┃ ┃ ┣ 📄Public.rmd
 ┃ ┃ ┣ 📄Step.1_KNN.R
 ┃ ┃ ┣ 📄Step.2_Logit.R
 ┃ ┃ ┣ 📄Step.3_LDA.R
 ┃ ┃ ┣ 📄Step.4_QDA.R
 ┃ ┃ ┗ 📄Step.99_Complete_Code.R
 ┃ ┗ 📂Sang
 ┗ 📄README.md
```

<p align="right">
<a href="https://github.com/theRealLeif/STAT387" target="_blank">
<img src="https://img.shields.io/github/last-commit/theRealLeif/STAT387?label=Last%20commit"/>
</a>
</p>

<h2 align="center">Reference</h2>

[1]: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
