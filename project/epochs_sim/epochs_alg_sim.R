###############################################################################
# This code simulate algebraic distributions normalizing the complete time
# series before the rotation and scale.
# Author: Anton J. Heckens
###############################################################################

# Libraries
library(MixMatrix)

# Parameters
# Number of companies
K = 2
# Degrees of freedom
dof = 10
# Shape parameters
l = (dof + K) / 2
m = (2 * l) - K - 2

# Out-diagonal values
smalc = 0.3
CorrMat1= matrix(smalc, K,K)
# Diagonal values
diag(CorrMat1)=1

## Number of data points in the time series
en = 5000000

set.seed(1)

# Data matrix
Part = rmatrixt(n=en, mean=matrix(0, K, 1), U=CorrMat1, V=matrix(m), df=dof)
# Take the values of the matrix we are interested
PartMat = Part[,,1:en]

# Multiply by the scaling factor
PartMatNORM = scale(t(PartMat), center=TRUE, scale=TRUE) * sqrt(((en)/(en-1)))

# Mean of the time series
print('Mean 1: ')
mean(PartMatNORM[,1])
print('Mean 2: ')
mean(PartMatNORM[,2])

# Variance of the time series
print('Var 1: ')
TEST1 = PartMatNORM[,1]
(1/length(TEST1) * sum((TEST1)^2))

print('Var 2: ')
TEST2 = PartMatNORM[,2]
(1/length(TEST2) * sum((TEST2)^2))

# Rotate and scale function
RotateScaleFunc = function(irun, returnsDataset, a) {

	returnCut = (returnsDataset[(0+irun):((a-1)+irun), ])
    corrMat = ( cor( returnCut ,  method = c("pearson")) )
	eigenList = eigen(corrMat, symmetric = TRUE,
                      only.values = FALSE, EISPACK = FALSE)
	LambdaEigenVal = eigenList[[1]]
	EigenVec = eigenList[[2]]
	squareLambda= 1/sqrt(LambdaEigenVal)
	RotScaledReturnVec = diag(squareLambda) %*% (t(EigenVec) %*% t(returnCut))

	return(list(RotScaledReturnVec))
}

# Epochs window lengths
epochs <- list(10, 25, 40, 55, 100)

for (a in epochs){

    # Normalized full time series
    b = PartMatNORM

    # For disjunct intervals:
    disjunct_Max = floor((length(b[, 1])) / a)

    # Rotation and scale of epochs
    system.time({
    ListAggregateReturn = lapply(((1:disjunct_Max) * a) - (a - 1),
                                 RotateScaleFunc, returnsDataset = b,  a = a)
    })

    # "r first element of list"
    # "https://stackoverflow.com/questions/20428742/select-first-element-of-nested-list"
    # Flatten (?)
    firstElement = lapply(ListAggregateReturn, `[[`, 1)

    # Aggregation of returns
    agg_returns = do.call("c" , firstElement)

    write.csv(agg_returns,
     sprintf('../data/epochs_sim/epochs_sim_algebraic_agg_dist_ret_market_data_long_win_%d_K_200.csv', a))
}