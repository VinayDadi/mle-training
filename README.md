# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To execute the script
- Create an environment from the provided `env.yml` file
```
conda env create -f env.yml
```

> Note: The name of the environment is `mle-dev` which can be seen in the name section of `env.yml` file

- Activate the environment once installation is sucessful
```
conda activate mle-dev
```

- Navigate to the directory where script is present and run the python script
```
python3 nonstandardcode.py
```
