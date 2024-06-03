//
// Created by Radu Chirila on 22/09/2021.
//

/*
int forward (int in1, int in2)
{
    return ((in1*w1 + in2*w2)>=t ? 1 : 0);
}
void training(float correct)
{
    float output = forward(in1,in2);
    w1 = w1 + learning_rate * (correct - output) * in1;
    w2 = w2 + learning_rate * (correct - output) * in2;
    t = t - learning_rate * (correct - output);
}
int main()
{
    int epochs = 280, batch = 4;
    FILE *input = fopen("train.txt", "r");
    int train[4][3];
    for (int i=0;i<=3;++i)
    {
        fscanf(input, "%d %d %d", &train[i][0],&train[i][1],&train[i][2]);
    }
    fclose(input);
    for(int i =0;i<epochs;++i)
    {
        for(int j=0;j<batch;++j)
        {
            in1 = train[j][0];
            in2 = train[j][1];
            training(train[j][2]);
        }
        printf("EPOCH: %d, W1: %f, W2: %f, T: %f \n \n",i, w1, w2, t);
    }
    //printf("0 0 %d \n0 1 %d \n1 0 %d \n1 1 %d\n", forward(0,0),forward(0,1), forward(1,0), forward(1,1));
    //printf("%d  %d", forward(30,1), forward(50,1));
    printf("%d", forward(1,1));
    return 0;
}*/