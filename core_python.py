## these fine contains core python algos and maths without the use of any libraries.
## this is to understand what is happening behind the scenes and how to implement them from scratch.
## this will help me understand the core concepts of python and programming in general.

##vectors and scalers 
##scalers are simple values like integers and floats. Vectors are a collection of scalers.
x = 5

##IN PYTHON vectors aare a collection of numbers. in python we use list or tuples to replresent vectors.

v1 = [1, 2, 3] # this is a vector represented as a list
v2 = [4, 5, 6] # this is another vector represented as a list

## VECTOR OPERATIONS

##addition of two vectors
##here v1 and v2 are two vectors represented as lists. 
# they need to be of the same length for addition to be possible. 
# the function will return a new vector which is the sum of the two input vectors.
def vector_add(v1,v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    return [v1[i] +  v2[i] for i in range(len(v1))]
## print (vector_add(v1,v2))

##       v1 + v2
##    [1 , 2, 3] + [4, 5, 6]
##       [1+4, 2+5, 3+6]
##      output  [5, 7, 9]

## MULTIPLICATION of two vector 
## dot product of two vectors
## the dot product of two vectors is the sum of the products of their corresponding elements. 
# the function will return a scalar value which is the dot product of the two input vectors.

def dot_product(v1,v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    return sum(v1[i] * v2[i] for i in range(len(v1)))

## print(dot_product(v1,v2))

## MATRIX 
##matrix is a collection of vectors. in python we can represent a matrix as a list of lists. 
# each inner list represents a row of the matrix.

m1 = [
    [1, 2], 
    [3, 4],
] # this is a matrix represented as a list of lists
m2 = [
    [5, 6], 
    [7, 8]
] # this is another matrix represented as a list of lists
##matrix operartions
##ADDITION of two matrices
## the function will return a new matrix which is the sum of the two input matrices.

##result = []

##for i in range(len(m1)):
##    row = []
##    for j in range(len(m1[0])):
##        row.append(m1[i][j] + m2[i][j])
##    result.append(row)
##
##print(result)


def matrix_addition(m1,m2):
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        raise ValueError("Matrices must be of the same dimensions")
    return [[m1[i][j] + m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]

## print(matrix_addition(m1,m2))

## MULTIPLICATION of two matrices
## the function will return a new matrix which is the product of the two input matrices.

def matrix_multiplication(m1,m2):
    if len(m1[0]) != len(m2):
        raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix")
    result = []
    for i in range(len(m1)):
        row = []
        for j in range(len(m2[0])):
            sum = 0 
            for k in range(len(m1[0])):
                sum += m1[i][k] * m2[k][j]
            row.append(sum)
        result.append(row)
    return result

print(matrix_multiplication(m1,m2))
