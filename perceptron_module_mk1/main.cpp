/*
 * Author: Radu Chirila
 * Date Created: 1.07.2021
 * Description: Neural network with backpropagation in C
 *
 * BEST Group, University of Glasgow
 */

/*
 * TODO:
 *  - modular dense definition of dense architecture
 *  - adaptive learning rate to minimize loss - look at loss functions
 *  - MAKE CODE READY FOR NUCLEO
 */
#include <cstdio>
#include <random>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
using namespace std;

//HYPERPARAMS
float lr = 0.0001;
int layers;
int batch=4;
float current_error=9;
int epochs = 1000;
ofstream err;
constexpr int FLOAT_MIN = 0;
constexpr int FLOAT_MAX = 1;

//PRIMITIVE GENERAL STRUCTURES
struct matrix{
    float *value;
    int row,col;
};
matrix inputs[4];
matrix expected[4];
struct network{
    matrix weights;
    matrix inputs;
    matrix output;
    //matrix expected;
    //short activation; // 0 relu, 1 sigmoid for now
}architecture[5];

//FUNCTION DEFINITIONS START
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
inline matrix read_mat(char *name)
{
    matrix A;
    ifstream fin;
    fin.open(name);
    fin>> A.row >> A.col;
    A.value = (float *) malloc(sizeof(float)*A.row*A.col);
    for(int i=0;i<A.row*A.col;++i)
    {
        fin >> *(A.value + i);
    }
    fin.close();
    return A;
} //read matrix from file (TESTING)
inline void print_mat(matrix A)
{
    for(int i=0;i<A.row*A.col;++i)
    {
        if(i%A.col == 0)
        {
            cout<<endl;
        }
        cout<<*(A.value+i)<< " ";
    }
}
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
inline matrix backprop(matrix expected, matrix output,matrix input)
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
    //error = (expected - output) * transfer_derivative(output), where transfer derivative is activation_back
    //error = weight_k * error_j * transfer_derivative(output)
}// can be recursive!!!!!
inline matrix forward_propagation(matrix w, matrix x)
{
    matrix out;
    out = matmul(w,x);
    for(int i=0;i<=out.col*out.row;++i)
    {
        *(out.value+i) = activation_relu(*(out.value+i));
    }
    return out;
} // can be recursive
inline void print_error(float b)
{

    err.open("error.txt", ios_base::app);
    err<<b<<endl;
    err.close();
}
inline float mean_error(int b)
{
    current_error = *(architecture[layers-1].output.value)-*(expected[b].value);
    current_error= abs(current_error);
    return current_error;
}
inline void train()
{
    for(int i=0;i<epochs;++i)
    {
        float mean_err=0;
        for(int j=0;j<batch;++j)
        {
            //forward pass
            for(int l=0;l<layers;++l)
            {
                    architecture[l+1].inputs = forward_propagation(matrix_transpose(architecture[l].weights),
                                                                     (l == 0 ? inputs[j] : architecture[l].inputs));
                    architecture[l].output = architecture[l + 1].inputs;
                if(l==layers-1)
                    architecture[l].output = forward_propagation(matrix_transpose(architecture[l].weights),architecture[l].inputs);
            }
            //backwards pass
            for(int l=layers-1;l>=0;--l) // goes 2 1 0 2 not updated
            {
                update_weights(architecture[l].weights, lr,
                               backprop((l==layers-1?expected[j]:architecture[l].output), architecture[l].output,
                                        (l==0?inputs[j]:architecture[l].inputs)), architecture[l].output);
            }
            mean_err+= mean_error(j);
        }
        print_error(mean_err/4);
    }
}
inline void randomize_weights()
{
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> distr(FLOAT_MIN, FLOAT_MAX);
    for(int i=0;i<layers;++i)
    {
        for(int j=0;j<architecture[i].weights.row*architecture[i].weights.col;++j)
        {
            *(architecture[i].weights.value+j)= distr(eng);//((float) rand()/(float)(RAND_MAX)*8.0);
        }
    }
}
//inline void init_architecture(char *file)
//{
//    ifstream fin;
//    fin.open(file);
//    fin>>layers;
//    int x,y;
//    for(int i=0;i<layers;++i)
//    {
//        if (i==0)
//        {
//            fin>>x;
//            architecture[i].weights = init_mat(x,x);
//            architecture[i].output= init_mat(x,1);
//        }
//        else
//        {
//            fin>>y;
//            architecture[i].weights = init_mat(x,y);
//            architecture[i].output = init_mat(y,1);
//            x=y;
//        }
//    }
//
//
//
//    randomize_weights();
//
//    ifstream fin2;
//    fin2.open("network_inputs.txt");
//    fin2>> batch>> inputs[0].row >> inputs[0].col;
//    for(int b=0;b<batch;++b) {
//        inputs[b].row = inputs[0].row;
//        inputs[b].col = inputs[0].col;
//        inputs[b].value = (float *) malloc(sizeof(float) * inputs[0].row * inputs[0].col);
//    }
//    for (int b=0;b<batch;++b){
//        for (int i = 0; i < inputs[b].row * inputs[b].col; ++i) {
//            fin2 >> *(inputs[b].value + i);
//        }
//    }
//    fin2.close();
//
//    ifstream fin3;
//    fin3.open("expected.txt");
//    fin3>> batch>> expected[0].row >> expected[0].col;
//    for(int b=0;b<batch;++b) {
//        expected[b].row = expected[0].row;
//        expected[b].col = expected[0].col;
//        expected[b].value = (float *) malloc(sizeof(float) * expected[0].row * expected[0].col);
//    }
//    for(int b=0;b<batch;++b) {
//        for (int i = 0; i < expected[b].row * expected[b].col; ++i) {
//            fin3 >> *(expected[b].value + i);
//        }
//    }
//    fin3.close();
//}
inline void init_architecture()
{
    layers=3;
    int x=6,y=6,z=1;
    architecture[0].weights = init_mat(x,x);
    architecture[0].output= init_mat(x,1);
    architecture[1].weights = init_mat(x,y);
    architecture[1].output = init_mat(y,1);
    architecture[2].weights = init_mat(y,z);
    architecture[2].output = init_mat(z,1);
    randomize_weights();
    batch = 3;
    inputs[0].row = 6;
    inputs[0].col = 1;
    for(int b=0;b<batch;++b) {
        inputs[b].row = inputs[0].row;
        inputs[b].col = inputs[0].col;
        inputs[b].value = (float *) malloc(sizeof(float) * inputs[0].row * inputs[0].col);
    }
    for (int b=0;b<batch;++b){
        for (int i = 0; i < inputs[b].row * inputs[b].col; ++i) {
            *(inputs[b].value + i)=1.0-0.5*b;
        }
    }

    batch = 3;
    expected[0].row = 1;
    expected[0].col = 1;
    for(int b=0;b<batch;++b) {
        expected[b].row = expected[0].row;
        expected[b].col = expected[0].col;
        expected[b].value = (float *) malloc(sizeof(float) * expected[0].row * expected[0].col);
    }
    for(int b=0;b<batch;++b) {
        for (int i = 0; i < expected[b].row * expected[b].col; ++i) {
            *(expected[b].value + i)=1-0.5*b;
        }
    }

}
int main()
{

//    init_architecture("architecture_reversed.txt");
init_architecture();
train();
    return 0;
}


/*
 * TODO:
 * - Implement network architecture (DONE)
 * - IMPLEMENT EXPONENTIAL LR ADJUSTMENTS (IN_PROGRESS)
 * - APPROPRIATE COMPILER GUARDS
 * - Optimize operations (binary?)
 *
 * _______________DONE ___________________
 * - Implement forward propagation (DONE)
 * - Implement training (DONE)
 * - Implement memory deallocation dynamic (DONE)
 * - ADJACENT MATRIX GRAPH BENEFITS??


 * - Test matmul (Done)
 * - Test addition (Done)
 * - Test substraction (Done)
 * - Test transpose (Done)
 * - Test scalar multiplication (Done)
 * - Test activation functions (Done)
 * - Plot activation functions (Done)
 * - IMPLEMENT KRONEKER MULTIPLICATION (DONE)
 * - TEST KRONEKER MULTIPLICATION (DONE)
 * - Test update weights (DONE)
 * - Test backpropagation (DOne)
 * - Test backprop + update (DONE)
 */

/*
 * OLD TESTING CODE
//    matrix A = read_mat("input.txt");
//    matrix B = read_mat("input2.txt");
//    matrix C = matmul(A,B);
//    matrix C = matrix_transpose(A);
//    matrix C = matrix_add(A,B);
//    matrix C = matrix_subtract(A,B);
//   matrix C = scalar_multiplication(A, 12);
//    print_mat(C);
//    ofstream out1, out2, out3, out4;
//    out1.open("output_activation_relu.dat");
//    out2.open("output_activation_relu_back.dat");
//    out3.open("output_activation_sigmoid.dat");
//    out4.open("output_activation_sigmoid_back.dat");
//    for(float i=-10.0;i<=10.0;i+=0.1)
//    {
//        out1<<i<<" "<<activation_relu(i)<<endl;
//        out2<<i<<" "<<activation_relu_back(i)<<endl;
//        out3<<i<<" "<<activation_sigmoid(i)<<endl;
//        out4<<i<<" "<<activation_sigmoid_back(i)<<endl;
//    }
//    out1.close();out2.close();out3.close();out4.close();
//    system("gnuplot -p -e \"plot 'output_activation_relu.dat' w l \"");
//    system("gnuplot -p -e \"plot 'output_activation_relu_back.dat' \"");
//    system("gnuplot -p -e \"plot 'output_activation_sigmoid.dat' \"");
//    system("gnuplot -p -e \"plot 'output_activation_sigmoid_back.dat' \"");
//    system("pause");
 */