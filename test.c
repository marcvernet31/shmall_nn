#include <stdio.h>
#include <stdlib.h>
#include "simple_neural_networks.h"

// Size of the layers
#define NUM_OF_FEATURES   	3  	// input values
#define NUM_OF_HID1_NODES		3	  // hidden nodes
#define NUM_OF_OUT_NODES 		1		// output classes


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
double w2[NUM_OF_OUT_NODES][NUM_OF_HID1_NODES] =    {{0.48, 0.73, 0.03}};

// Predicted values
double yhat[1][NUM_OF_OUT_NODES];
double yhat_eg[NUM_OF_OUT_NODES];	// Predicted yhat

// Training data
double train_x[1][NUM_OF_FEATURES];				// Training data after normalization
double train_y[1][NUM_OF_OUT_NODES] = {{1}};  	// The expected (training) y values

double raw_x[1][NUM_OF_FEATURES] = {{23.0, 40.0, 100.0}};	// temp, hum, air_q input values

// Lab 4.1 - backpropagation
double dA1[1][NUM_OF_HID1_NODES] = {{0, 0, 0}};
double dA2[1][1] = {{0}};

double dZ1[1][NUM_OF_HID1_NODES] = {{0, 0, 0}};
double dZ2[1][1] = {{0}};

double dW1[NUM_OF_HID1_NODES][NUM_OF_FEATURES] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
double dW2[NUM_OF_OUT_NODES][NUM_OF_HID1_NODES] = {{0, 0, 0}};

double db1[NUM_OF_HID1_NODES] = {0, 0, 0};
double db2[NUM_OF_OUT_NODES] = {0};
double W2_T[NUM_OF_HID1_NODES][NUM_OF_OUT_NODES] = {{0},{0},{0}};


void init(){
  normalize_data_2d(NUM_OF_FEATURES,1, raw_x, train_x);	// Data normalization
  weightsB_zero_initialization(b1, NUM_OF_HID1_NODES);
  weightsB_zero_initialization(b2, NUM_OF_OUT_NODES);
}


void iteration(int i){

  linear_forward_nn(train_x, NUM_OF_FEATURES, z1[0], NUM_OF_HID1_NODES, w1, b1);
  vector_relu(z1[0],a1,NUM_OF_HID1_NODES);
  linear_forward_nn(a1, NUM_OF_HID1_NODES, z2[0], NUM_OF_OUT_NODES, w2, b2);
  vector_sigmoid(z2[0],yhat[0], NUM_OF_OUT_NODES);

  double cost = compute_cost(1, yhat, train_y);
  printf("%d cost: %f\r\n", i, cost);
  printf("--------- \n");

  matrix_matrix_sub(NUM_OF_OUT_NODES, NUM_OF_OUT_NODES, yhat, train_y, dZ2);
  linear_backward(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES,  1, dZ2, a1, dW2, db2);
  matrix_transpose(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, w2, W2_T);

  matrix_matrix_multiplication(NUM_OF_HID1_NODES, NUM_OF_OUT_NODES, NUM_OF_OUT_NODES, W2_T, dZ2, dA1);

  relu_backward(1, NUM_OF_HID1_NODES, dA1, z1, dZ1);

  linear_backward(NUM_OF_HID1_NODES, NUM_OF_FEATURES, 1, dZ1, train_x, dW1, db1);

  weights_update(NUM_OF_HID1_NODES, NUM_OF_FEATURES, learning_rate, dW1, w1);

  weights_update(NUM_OF_HID1_NODES, NUM_OF_FEATURES, learning_rate, db1, b1);

  weights_update(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, learning_rate, dW2, w2);

  weights_update(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, learning_rate, db2, b2);


}

int main(){
  init();
  for(int i = 0; i < 100; ++i) iteration(i);
}
