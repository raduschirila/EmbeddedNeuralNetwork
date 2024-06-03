#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
using namespace std;
float lr=0.0015; // FOR TESTING
ofstream dout;
struct matrix{
    float *value;
    int row,col;
};
struct network{
    matrix weights;
    matrix inputs;
    matrix output;
    matrix expected;
}architecture[2];
inline matrix init_mat(int row, int col)
{
    matrix x;
    x.row = row;
    x.col = col;
    x.value = new float[x.row * x.col];
    return x;
}
inline void delete_mat(matrix A)
{
    delete[] A.value;
}
inline matrix matmul(matrix A,matrix B)
{
    matrix result;
    if(A.col != B.row)
    {
        printf("Check matrix dimensions!\n");
        return result;
    }
    else {
        float sum;
        //result = init_mat(A.row, B.col);
        result.row = A.row;
        result.col = B.col;
        result.value = new float[result.row * result.col];
        for (int row = 0; row < A.row; ++row)
        {
            for (int col = 0; col < B.col; ++col)
            {
                sum = 0;
                for (int it = 0; it < B.row; ++it)
                {
                    sum += (*(A.value + A.col*row + it) * *(B.value +B.col * it + col));
                }
                *(result.value + result.col * row + col) = sum;
            }
        }
        return result;
    }
}
inline matrix matrix_add(matrix A, matrix B)
{
    matrix result;
    if(A.row !=B.row || A.col != B.col)
    {
        printf("Incompatible matrices!");
        return result;
    }
    else
    {
        result.row = A.row;result.col = A.col;
        result.value = new float[result.row * result.col];
        //result = init_mat(A.row, A.col);
        for(int i=0;i<=A.row*A.col;++i)
        {
            *(result.value+i) = *(A.value+i) + *(B.value+i);
        }
        return result;
    }
}
inline matrix matrix_subtract(matrix A, matrix B)
{
    matrix result;
    if(A.row !=B.row || A.col != B.col)
    {
        printf("Incompatible matrices!");
        return result;
    }
    else
    {
        result.row = A.row;result.col = A.col;
        result.value = new float[result.row * result.col];
        for(int i=0;i<=A.row*A.col;++i)
        {
            *(result.value+i) = *(A.value+i) - *(B.value+i);
        }
        return result;
    }
}
inline matrix matrix_transpose(matrix A)
{
    if(A.row == 1 || A.col == 1)
    {
        A.row ^= A.col ^= A.row ^= A.col;
        return A;
    }
    else {
        matrix result;
        result.row = A.row;result.col = A.col;
        result.value = new float[result.row * result.col];
        for (int i = 0; i < A.row; ++i) {
            for (int j = 0; j < A.col; ++j) {
                *(result.value + A.row * i + j) = *(A.value + i + A.col * j);
            }
        }
        result.row ^= result.col ^= result.row ^= result.col;
        return result;
    }
}
inline matrix scalar_multiplication(matrix A, float x)
{
    matrix result;
    result.row = A.row; result.col = A.col;
    result.value = new float[result.row * result.col];
    for(int i=0;i<=A.row*A.col;++i)
    {
        *(result.value+i) = *(A.value+i) * x;
    }
    return result;
}
inline matrix matmul_elementwise(matrix A, matrix B)
{
    matrix result;
    if(A.row !=B.row || A.col != B.col)
    {
        printf("Incompatible matrices!");
        return result;
    }
    else
    {
        result.row = A.row;result.col = A.col;
        result.value = new float[result.row * result.col];
        for(int i=0;i<=A.row*A.col;++i)
        {
            *(result.value+i) = *(A.value+i) * *(B.value+i);
        }
        return result;
    }
}// Hadamard product function


inline float activation_relu(float x)
{
    return (x<0 ? 0 : x);
}
inline float activation_relu_back(float x)
{
    return (x<0 ? 0:1);
}
inline float activation_sigmoid(float x)
{
    return 1.0 / (1.0 +exp(-x));
}
inline float activation_sigmoid_back(float x)
{
    return x*(1.0-x);
}
inline matrix matmul_kroneker(matrix A, matrix B)
{
    matrix result;
    int startRow, startCol;
    result.row = A.row*B.row;
    result.col = A.col*B.col;
    result.value = new float[result.row * result.col];
    for(int i=0;i<A.row;i++){
        for(int j=0;j<A.col;j++){
            startRow = i*B.row;
            startCol = j*B.col;
            for(int k=0;k<B.row;k++){
                for(int l=0;l<B.col;l++){
                    *(result.value + result.col*(startRow + k) + startCol + l) = *(A.value + A.col * i + j) * *(B.value + B.col*k + l);
                }
            }
        }
    }
    return result;
}
inline void update_weights(matrix &w, float lr, matrix delta, matrix output)
{
    w = matrix_add(w, scalar_multiplication(matmul_kroneker(matrix_transpose(output), delta),lr));
}
inline matrix backprop(matrix w, matrix expected, matrix output,matrix input)
{
    matrix error,delta;
    delta.row = input.row;
    delta.col = input.col;
    delta.value = new float[delta.row*delta.col];
    for(int i=0;i<delta.row*delta.col;++i)
    {
        *(delta.value+i) = activation_relu_back(*(output.value + i));
    }
    error = matrix_subtract(expected,output);
    delta = (delta.col == 1 ? scalar_multiplication(delta,*(error.value)): matmul_elementwise(error, delta));
    return delta;
}// can be recursive!!!!!
inline matrix forward_propagation(matrix w, matrix x)
{
    return matmul(w,x);
} // can be recursive
inline void print_error()
{

    printf("Error: %f\n",*(architecture[1].output.value)-*(architecture[1].expected.value));
}
inline void train(int epochs)
{
    for(int i=0;i<epochs;++i)
    {
        printf("Epoch: %d ",i);
        lr = (i<=6 ? 0.001: 0.0015);
        architecture[1].inputs = forward_propagation(matrix_transpose(architecture[0].weights), architecture[0].inputs);
        architecture[0].output = architecture[1].inputs; // basically same layer anyway
        architecture[1].output = forward_propagation(matrix_transpose(architecture[1].weights), architecture[1].inputs);
        update_weights(architecture[1].weights, lr, backprop(architecture[1].weights, architecture[1].expected, architecture[1].output, architecture[1].inputs), architecture[1].output);
        update_weights(architecture[0].weights, lr, backprop(architecture[0].weights, architecture[0].output, architecture[0].output, architecture[0].inputs),architecture[0].output);
        print_error();

    }
}
inline void randomize_weights()
{
    for(int i=0;i<2;++i)
    {
        for(int j=0;j<architecture[i].weights.row*architecture[i].weights.col;++j)
        {
            *(architecture[i].weights.value+j)= ((float) rand()/(float)(RAND_MAX)*2.0);
        }
    }
}
inline void init_architecture()
{
    int x=4,y=1;
    architecture[0].weights = init_mat(x,x);
    architecture[0].inputs.row = 4;
    architecture[0].inputs.col = 1;
    architecture[0].inputs.value = new float[architecture[0].inputs.col * architecture[0].inputs.row];
    *(architecture[0].inputs.value) = 2;
    *(architecture[0].inputs.value + 1) = 3;
    *(architecture[0].inputs.value + 2) = 4;
    *(architecture[0].inputs.value + 3) = 1;
    architecture[0].output= init_mat(x,1);

    architecture[1].weights = init_mat(x,y);
    architecture[1].output = init_mat(x,1);
    architecture[1].expected.row = 1;
    architecture[1].expected.col = 1;
    architecture[1].expected.value = new float[architecture[1].expected.col * architecture[1].expected.row];
    *(architecture[1].expected.value) = 3;
    randomize_weights();
}
int main()
{
    init_architecture();
    train(12);
    system("pause");
    return 0;
}

