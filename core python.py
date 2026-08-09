## these fine contains core python algos and maths without the use of any libraries.
## this is to understand what is happening behind the scenes and how to implement them from scratch.
## this will help me understand the core concepts of python and programming in general.

##vectors and scalers 
##scalers are simple values like integers and floats. Vectors are a collection of scalers.
x = 5

##IN PYTHON vectors aare a collection of numbers. in python we use list or tuples to replresent vectors.

x = [1, 2, 3, 4, 5] # this is a vector represented as a list

## VECTOR OPERATIONS

##addition of two vectors
##here v1 and v2 are two vectors represented as lists. they need to be of the same length for addition to be possible. the function will return a new vector which is the sum of the two input vectors.
def vector_add(v1,v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    return [v1[i] +  v2[i] for i in range(len(v1))]

##       v1 + v2
##    [1 , 2, 3] + [4, 5, 6]
##       [1+4, 2+5, 3+6]
##      output  [5, 7, 9]



