# MLJTime
An [MLJ](https://github.com/alan-turing-institute/MLJ.jl) compatible Julia toolbox for machine learning with
time series.

[![Build Status](https://travis-ci.com/alan-turing-institute/MLJTime.jl.svg?branch=master)](https://travis-ci.com/alan-turing-institute/MLJTime.jl)
[![Coverage](http://codecov.io/github/alan-turing-institute/MLJTime.jl/coverage.svg?branch=master)](http://codecov.io/github/alan-turing-institute/MLJTime.jl?branch=master)


## Installation
To install MLJTime.jl, launch Julia and run:
```julia
]add "https://github.com/alan-turing-institute/MLJTime.jl.git"
```
MLJTime.jl requires Julia version 1.0 or greater.

## Quickstart 
```julia
using MLJTime

# load data
X, y = ts_dataset("Chinatown")

# split data into training and test set
train, test = partition(eachindex(y), 0.7, shuffle=true, rng=1234) #70:30 split
X_train, y_train = X[train], y[train];
X_test, y_test = X[test], y[test];

# train model
model = TimeSeriesForestClassifier(n_trees=3)
mach = machine(model, matrix(X_train), y_train)
fit!(mach)

# make predictions
y_pred = predict_mod(mach, X_test)
```

## Documentation
To find out more, check out our:

* [Blog post](https://nextjournal.com/aa25desh)
* [Tutorial](https://github.com/aa25desh/MLJTimeTutorials.jl)

## Future work
In future work, we want to add:

* Support for multivariate time series,
* Shapelet based classification algorithms,
* Enhancements to KNN (KDTree and BallTree algorithms),
* Forecasting framework.

## How contribute
* If you are interested, please raise an issue or get in touch with the MLJTime team on slack. 

## About the project
This project was originally developed as part of the Google Summer of Code 2020 with the support of the Julia community and my mentors [Sebastian Vollmer](https://warwick.ac.uk/fac/sci/maths/people/staff/vollmer/) and [Markus Löning](https://github.com/mloning).

Active maintainers: 
* [Aadesh Deshmukh](https://github.com/aa25desh)
* Markus Löning
* Sebastian Vollmer

