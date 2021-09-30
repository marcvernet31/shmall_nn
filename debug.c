#include <stdio.h>
#include <stdlib.h>
#include "simple_neural_networks.h"

// Size of the layers
#define NUM_OF_FEATURES   	3  	// input values
#define NUM_OF_HID1_NODES		3	  // hidden nodes
#define NUM_OF_OUT_NODES 		2		// output classes

double learning_rate = 0.01;

/*Input layer to hidden layer*/
double a1[1][NUM_OF_HID1_NODES];	// activation function
double b1[NUM_OF_HID1_NODES];			// bias
double z1[1][NUM_OF_HID1_NODES];	// output vector

// Input layer to hidden layer weight matrix
double w1[NUM_OF_HID1_NODES][NUM_OF_FEATURES] =    {{0.25, 0.5,   0.05},   	 //hid[0]
																										{0.8,  0.82,  0.3 },     //hid[1]
																										{0.5,  0.45,  0.19}};   //hid[2]

/*Hidden layer to output layer*/
double b2[NUM_OF_OUT_NODES];
double z2[1][NUM_OF_OUT_NODES];	// Predicted output vector

// Hidden layer to output layer weight matrix
double w2[NUM_OF_OUT_NODES][NUM_OF_HID1_NODES] = {{0.25, 0.5,   0.05},   	 //hid[0]
																										{0.8,  0.82,  0.3 }     //hid[1]
																	};

// Predicted values
double yhat[1][NUM_OF_OUT_NODES];
double yhat_eg[NUM_OF_OUT_NODES];	// Predicted yhat

// Training data
double train_x[1][NUM_OF_FEATURES];				// Training data after normalization
double train_y[1][NUM_OF_OUT_NODES] = {{0, 1}};  	// The expected (training) y values



int main(void) {
	// Raw training data
	double raw_x[1][NUM_OF_FEATURES] = {{23.0, 40.0, 100.0}};	// temp, hum, air_q input values

	normalize_data_2d(NUM_OF_FEATURES,1, raw_x, train_x);	// Data normalization
	printf("train_x \n");
	matrix_print(1, NUM_OF_FEATURES, train_x);

	weightsB_zero_initialization(b1, NUM_OF_HID1_NODES);
	weightsB_zero_initialization(b2, NUM_OF_OUT_NODES);


	// Lab 3.1
	// Warning !!!
	linear_forward_nn(train_x, NUM_OF_FEATURES, z1[0], NUM_OF_HID1_NODES, w1, b1);
	printf("Output vector (Z1_1): %f\n", z1[0][0]);
	printf("Output vector (Z1_2): %f\n", z1[0][1]);
	printf("Output vector (z1_3): %f\r\n", z1[0][2]);

	// Warning !!!
	vector_relu(z1[0],a1,NUM_OF_HID1_NODES);
	printf("relu_a \n");
	matrix_print(1, NUM_OF_HID1_NODES, a1);

	// Warning !!!
	linear_forward_nn(a1, NUM_OF_HID1_NODES, z2[0], NUM_OF_OUT_NODES, w2, b2);
	printf("Output vector (Z2): \n");
  matrix_print(1, NUM_OF_OUT_NODES, z2);


	// Warning !!!
	vector_sigmoid(z2[0],yhat[0], NUM_OF_OUT_NODES);
  printf("yhat: \n");
  matrix_print(1, NUM_OF_OUT_NODES, yhat);

	double cost = compute_cost(1, NUM_OF_OUT_NODES, yhat, train_y);
	printf("cost:  %f\r\n", cost);


	// Lab 4.1 - backpropagation
	double dA1[1][NUM_OF_HID1_NODES];
	double dA2[1][1];

	double dZ1[1][NUM_OF_HID1_NODES];
	double dZ2[1][NUM_OF_OUT_NODES];

	double dW1[NUM_OF_HID1_NODES][NUM_OF_FEATURES];
	double dW2[NUM_OF_OUT_NODES][NUM_OF_HID1_NODES];

	double db1[NUM_OF_HID1_NODES];
	double db2[NUM_OF_OUT_NODES];


	// Output layer

	//dZ2 = A2-Y = yhat - y
	//TODO: calculate dZ2, use matrix_matrix_sub() function
	matrix_matrix_sub(NUM_OF_OUT_NODES, NUM_OF_OUT_NODES, yhat, train_y, dZ2);


	printf("dZ2 \n");
	matrix_print(NUM_OF_OUT_NODES, NUM_OF_OUT_NODES, dZ2);


	// TODO: Calculate linear backward for output layer, use linear_backward() function
	//check for formula on slide 31 (lecture 5)
	linear_backward(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES,  1, dZ2, a1, dW2, db2);


	printf("dW2 \n");
	matrix_print(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, dW2);

	printf("db2 \n");
	matrix_print(NUM_OF_OUT_NODES, 1, db2);


	double W2_T[NUM_OF_HID1_NODES][NUM_OF_OUT_NODES];


	// TODO: Make matrix transpose for output layer, use matrix_transpose() function
	matrix_transpose(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, w2, W2_T);

	printf("W2_T \n");
	matrix_print(NUM_OF_HID1_NODES, NUM_OF_OUT_NODES, W2_T);


	// TODO: Make matrix matrix multiplication; use matrix_matrix_multiplication() function
	// Check for formula on slide 31 (lecture 5)
	matrix_matrix_multiplication(NUM_OF_HID1_NODES, NUM_OF_OUT_NODES, NUM_OF_OUT_NODES, W2_T, dZ2, dA1);

	printf("dA1 \n");
	matrix_print(1, NUM_OF_HID1_NODES, dA1);


	// Input layer //

	// TODO: Calculate relu backward for hidden layer, use relu_backward() function
	// Check for formula on slide 31 (lecture 5)
	relu_backward(1, NUM_OF_HID1_NODES, dA1, z1, dZ1);

	printf("dZ1 \n");
	matrix_print(1, NUM_OF_HID1_NODES, dZ1);



	// TODO: Calculate linear backward for hidden layer, use linear_backward() function
	// Check for formula on slide 31 (lecture 5)
	linear_backward(NUM_OF_HID1_NODES, NUM_OF_FEATURES, 1, dZ1, train_x, dW1, db1);


	// UPDATE PARAMETERS //

	// W1 = W1 - learning_rate * dW1
	// TODO: update weights for W1, use weights_update() function
	weights_update(NUM_OF_HID1_NODES, NUM_OF_FEATURES, learning_rate, dW1, w1);


	printf("updated W1  \n");
	matrix_print( NUM_OF_HID1_NODES, NUM_OF_FEATURES, w1);


	// b1 = b1 - learning_rate * db1
	// TODO: update bias for b1, use weights_update() function
	weights_update(NUM_OF_HID1_NODES, NUM_OF_FEATURES, learning_rate, db1, b1);


	printf("updated b1  \n");
	matrix_print(NUM_OF_HID1_NODES, 1, b1);

	// W2 = W2 - learning_rate * dW2
	// TODO: update weights for W2, use weights_update() function
	weights_update(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, learning_rate, dW2, w2);

	printf("updated W2  \n");
	matrix_print( NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, w2);

	// b2 = b2 - learning_rate * db2
	// TODO: update bias for b2, use weights_update() function
	weights_update(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, learning_rate, db2, b2);


	printf("updated b2  \n");
	matrix_print( NUM_OF_OUT_NODES, 1, b2);

}
