#include <stdio.h>
#include <stdlib.h>
#include "simple_neural_networks.h"

// Size of the layers
#define NUM_OF_FEATURES   	4  	// input values
#define NUM_OF_HID1_NODES		3	  // hidden nodes
#define NUM_OF_OUT_NODES 		3		// output classes
#define CASES               50 //
#define EPOCH               50


double learning_rate = 0.01;

/*Input layer to hidden layer*/
double a1[1][NUM_OF_HID1_NODES];	// activation function
double b1[NUM_OF_HID1_NODES];			// bias
double z1[1][NUM_OF_HID1_NODES];	// output vector


// Training data
double train_x[1][NUM_OF_FEATURES];				// Training data after normalization
double train_y[1][NUM_OF_OUT_NODES] = {{0, 1, 0},{1, 0, 0},{0, 0, 1},{0, 0, 1},{0, 1, 0},{0, 1, 0},{0, 0, 1},{1, 0, 0},{1, 0, 0},{0, 0, 1},{1, 0, 0},{0, 1, 0},{0, 0, 1},{0, 0, 1},{0, 1, 0},{1, 0, 0},{0, 0, 1},{1, 0, 0},{1, 0, 0},{1, 0, 0},{0, 1, 0},{0, 1, 0},{0, 0, 1},{0, 0, 1},{1, 0, 0},{1, 0, 0},{0, 0, 1},{0, 1, 0},{0, 1, 0},{0, 0, 1},{0, 1, 0},{0, 0, 1},{0, 1, 0},{1, 0, 0},{0, 1, 0},{0, 1, 0},{0, 0, 1},{0, 0, 1},{1, 0, 0},{1, 0, 0},{1, 0, 0},{0, 0, 1},{0, 1, 0},{1, 0, 0},{1, 0, 0},{0, 0, 1},{0, 1, 0},{0, 0, 1},{0, 1, 0},{0, 0, 1},{0, 1, 0},{1, 0, 0},{0, 0, 1},{1, 0, 0},{0, 1, 0},{0, 1, 0},{0, 1, 0},{0, 0, 1},{0, 0, 1},{0, 1, 0},{0, 0, 1},{0, 1, 0},{1, 0, 0},{0, 1, 0},{0, 1, 0},{1, 0, 0},{0, 1, 0},{1, 0, 0},{0, 0, 1},{0, 0, 1},{0, 1, 0},{0, 1, 0},{1, 0, 0},{0, 1, 0},{0, 0, 1},{0, 1, 0},{1, 0, 0},{1, 0, 0},{1, 0, 0},{0, 0, 1},{0, 0, 1},{1, 0, 0},{0, 0, 1},{0, 0, 1},{0, 0, 1},{1, 0, 0},{0, 1, 0},{0, 0, 1},{0, 1, 0},{0, 0, 1},{0, 1, 0},{1, 0, 0},{0, 0, 1},{1, 0, 0},{1, 0, 0},{0, 1, 0},{0, 0, 1},{1, 0, 0},{1, 0, 0},{0, 0, 1},{0, 0, 1},{0, 0, 1},{0, 1, 0},{0, 0, 1},{0, 0, 1},{1, 0, 0},{1, 0, 0},{1, 0, 0},{0, 1, 0},{0, 0, 1},{1, 0, 0},{1, 0, 0},{0, 0, 1},{0, 1, 0},{1, 0, 0},{0, 0, 1},{1, 0, 0},{0, 1, 0},{0, 1, 0},{0, 1, 0},{0, 1, 0},{1, 0, 0},{0, 0, 1},{1, 0, 0},{1, 0, 0},{1, 0, 0},{0, 1, 0},{1, 0, 0},{1, 0, 0},{0, 1, 0},{0, 0, 1},{0, 1, 0},{1, 0, 0},{0, 1, 0},{0, 0, 1},{0, 0, 1},{1, 0, 0},{0, 0, 1},{0, 0, 1},{1, 0, 0},{0, 1, 0},{0, 1, 0},{0, 0, 1},{0, 1, 0},{0, 1, 0},{0, 1, 0},{0, 0, 1},{1, 0, 0},{0, 1, 0},{1, 0, 0}};

double raw_x[1][NUM_OF_FEATURES] = {{6.3,2.5,4.9,1.5,},{5.2,4.1,1.5,0.1,},{6.9,3.2,5.7,2.3,},{7.4,2.8,6.1,1.9,},{5.9,3.0,4.2,1.5,},{4.9,2.4,3.3,1.0,},{6.8,3.0,5.5,2.1,},{5.0,3.5,1.3,0.3,},{5.2,3.4,1.4,0.2,},{7.7,2.6,6.9,2.3,},{5.1,3.8,1.6,0.2,},{5.6,3.0,4.1,1.3,},{6.7,3.3,5.7,2.5,},{5.8,2.7,5.1,1.9,},{5.9,3.2,4.8,1.8,},{4.6,3.2,1.4,0.2,},{7.7,2.8,6.7,2.0,},{4.9,3.1,1.5,0.1,},{4.9,3.0,1.4,0.2,},{4.7,3.2,1.6,0.2,},{6.3,2.3,4.4,1.3,},{5.7,3.0,4.2,1.2,},{5.9,3.0,5.1,1.8,},{6.5,3.0,5.8,2.2,},{4.8,3.4,1.6,0.2,},{5.4,3.4,1.7,0.2,},{6.3,3.3,6.0,2.5,},{5.7,2.9,4.2,1.3,},{5.5,2.4,3.8,1.1,},{5.8,2.8,5.1,2.4,},{6.0,2.2,4.0,1.0,},{7.2,3.0,5.8,1.6,},{5.7,2.8,4.5,1.3,},{4.6,3.6,1.0,0.2,},{5.6,3.0,4.5,1.5,},{5.0,2.0,3.5,1.0,},{6.9,3.1,5.1,2.3,},{6.7,2.5,5.8,1.8,},{5.3,3.7,1.5,0.2,},{4.8,3.1,1.6,0.2,},{4.4,2.9,1.4,0.2,},{6.2,2.8,4.8,1.8,},{6.0,2.7,5.1,1.6,},{5.0,3.4,1.6,0.4,},{5.1,3.5,1.4,0.2,},{6.3,2.7,4.9,1.8,},{6.9,3.1,4.9,1.5,},{6.3,2.5,5.0,1.9,},{5.6,2.7,4.2,1.3,},{7.1,3.0,5.9,2.1,},{6.1,2.8,4.0,1.3,},{5.0,3.2,1.2,0.2,},{6.7,3.3,5.7,2.1,},{4.9,3.1,1.5,0.1,},{5.7,2.6,3.5,1.0,},{5.2,2.7,3.9,1.4,},{5.7,2.8,4.1,1.3,},{6.7,3.0,5.2,2.3,},{6.5,3.0,5.5,1.8,},{6.6,3.0,4.4,1.4,},{7.9,3.8,6.4,2.0,},{5.4,3.0,4.5,1.5,},{4.3,3.0,1.1,0.1,},{5.5,2.4,3.7,1.0,},{6.4,2.9,4.3,1.3,},{5.2,3.5,1.5,0.2,},{6.8,2.8,4.8,1.4,},{5.5,3.5,1.3,0.2,},{5.7,2.5,5.0,2.0,},{6.4,2.8,5.6,2.1,},{5.6,2.9,3.6,1.3,},{5.5,2.5,4.0,1.3,},{4.4,3.2,1.3,0.2,},{6.0,3.4,4.5,1.6,},{6.5,3.0,5.2,2.0,},{6.3,3.3,4.7,1.6,},{4.5,2.3,1.3,0.3,},{5.0,3.4,1.5,0.2,},{4.8,3.0,1.4,0.1,},{6.1,3.0,4.9,1.8,},{6.5,3.2,5.1,2.0,},{5.4,3.7,1.5,0.2,},{7.3,2.9,6.3,1.8,},{7.2,3.6,6.1,2.5,},{7.2,3.2,6.0,1.8,},{5.0,3.0,1.6,0.2,},{6.1,2.9,4.7,1.4,},{7.7,3.0,6.1,2.3,},{7.0,3.2,4.7,1.4,},{6.3,2.9,5.6,1.8,},{6.7,3.1,4.4,1.4,},{5.1,3.7,1.5,0.4,},{6.1,2.6,5.6,1.4,},{5.1,3.8,1.5,0.3,},{4.6,3.1,1.5,0.2,},{5.8,2.7,4.1,1.0,},{6.4,3.1,5.5,1.8,},{4.8,3.4,1.9,0.2,},{4.9,3.1,1.5,0.1,},{6.4,2.7,5.3,1.9,},{6.4,2.8,5.6,2.2,},{6.2,3.4,5.4,2.3,},{6.2,2.9,4.3,1.3,},{6.0,3.0,4.8,1.8,},{6.8,3.2,5.9,2.3,},{5.7,4.4,1.5,0.4,},{5.0,3.5,1.6,0.6,},{5.7,3.8,1.7,0.3,},{6.7,3.1,4.7,1.5,},{6.3,2.8,5.1,1.5,},{4.7,3.2,1.3,0.2,},{5.1,3.4,1.5,0.2,},{6.0,2.2,5.0,1.5,},{6.2,2.2,4.5,1.5,},{5.0,3.3,1.4,0.2,},{6.7,3.1,5.6,2.4,},{5.4,3.4,1.5,0.4,},{5.1,2.5,3.0,1.1,},{5.5,2.3,4.0,1.3,},{6.1,2.8,4.7,1.2,},{5.8,2.7,3.9,1.2,},{4.8,3.0,1.4,0.3,},{5.8,2.7,5.1,1.9,},{4.6,3.4,1.4,0.3,},{5.1,3.5,1.4,0.3,},{5.1,3.8,1.9,0.4,},{6.7,3.0,5.0,1.7,},{5.5,4.2,1.4,0.2,},{4.4,3.0,1.3,0.2,},{5.8,2.6,4.0,1.2,},{6.9,3.1,5.4,2.1,},{6.4,3.2,4.5,1.5,},{5.4,3.9,1.3,0.4,},{6.1,3.0,4.6,1.4,},{6.3,3.4,5.6,2.4,},{5.6,2.8,4.9,2.0,},{5.1,3.3,1.7,0.5,},{6.4,3.2,5.3,2.3,},{4.9,2.5,4.5,1.7,},{5.8,4.0,1.2,0.2,},{5.5,2.6,4.4,1.2,},{6.5,2.8,4.6,1.5,},{7.6,3.0,6.6,2.1,},{6.0,2.9,4.5,1.5,},{6.6,2.9,4.6,1.3,},{5.6,2.5,3.9,1.1,},{7.7,3.8,6.7,2.2,},{5.0,3.6,1.4,0.2,},{5.0,2.3,3.3,1.0,},{5.4,3.9,1.7,0.4,}};



/*
// Input layer to hidden layer weight matrix
double w1[NUM_OF_HID1_NODES][NUM_OF_FEATURES] =    {{0.25, 0.5,   0.05},   	 //hid[0]
																										{0.8,  0.82,  0.3 },     //hid[1]
																										{0.5,  0.45,  0.19}};   //hid[2]

// Hidden layer to output layer weight matrix
double w2[NUM_OF_OUT_NODES][NUM_OF_HID1_NODES] =    {{0.48, 0.73, 0.03}};
*/
double w1[NUM_OF_HID1_NODES][NUM_OF_FEATURES];
double w2[NUM_OF_OUT_NODES][NUM_OF_HID1_NODES];

/*Hidden layer to output layer*/
double b2[NUM_OF_OUT_NODES];
double z2[1][NUM_OF_OUT_NODES];	// Predicted output vector

// Predicted values
double yhat[1][NUM_OF_OUT_NODES];
double yhat_eg[NUM_OF_OUT_NODES];	// Predicted yhat

/*
// Backpropagation
double dA1[1][NUM_OF_HID1_NODES] = {{0, 0, 0}};
double dA2[1][1] = {{0}};

double dZ1[1][NUM_OF_HID1_NODES] = {{0, 0, 0}};
double dZ2[1][1] = {{0}};

double dW1[NUM_OF_HID1_NODES][NUM_OF_FEATURES] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
double dW2[NUM_OF_OUT_NODES][NUM_OF_HID1_NODES] = {{0, 0, 0}};

double db1[NUM_OF_HID1_NODES] = {0, 0, 0};
double db2[NUM_OF_OUT_NODES] = {0};
double W2_T[NUM_OF_HID1_NODES][NUM_OF_OUT_NODES] = {{0},{0},{0}};
*/

double dA1[1][NUM_OF_HID1_NODES];
// double dA2[1][1];
double dA2[1][NUM_OF_OUT_NODES]; //--

double dZ1[1][NUM_OF_HID1_NODES];
// double dZ2[1][1];
double dZ2[1][NUM_OF_OUT_NODES]; //--

double dW1[NUM_OF_HID1_NODES][NUM_OF_FEATURES];
double dW2[NUM_OF_OUT_NODES][NUM_OF_HID1_NODES];

double db1[NUM_OF_HID1_NODES];
double db2[NUM_OF_OUT_NODES];
double W2_T[NUM_OF_HID1_NODES][NUM_OF_OUT_NODES];

void init(){
  normalize_data_2d(NUM_OF_FEATURES,1, raw_x, train_x);	// Data normalization

	weights_random_initialization( NUM_OF_HID1_NODES, NUM_OF_FEATURES, w1);
	weights_random_initialization( NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, w2);

  weightsB_zero_initialization(b1, NUM_OF_HID1_NODES);
  weightsB_zero_initialization(b2, NUM_OF_OUT_NODES);
}

int accuracy = 0;

int checkPrediction(double yhat[NUM_OF_FEATURES], double train_y[NUM_OF_FEATURES]){
  int yhat_i = 0;
  int yhat_max = 0;
  int train_y_i = 0;
  int train_y_max = 0;
  for(int i = 0; i < NUM_OF_FEATURES; ++i){
    if(yhat[i] > yhat_max){
      yhat_i = i;
      yhat_max = yhat[i];
    }
    if(train_y[i] > train_y_max){
      train_y_i = i;
      train_y_max = train_y[i];
    }
  }
  if(train_y_i == yhat_i) return 1;
  return 0;
}

void forward(double raw_x[NUM_OF_FEATURES], double train_y[NUM_OF_FEATURES]){
	linear_forward_nn(train_x, NUM_OF_FEATURES, z1[0], NUM_OF_HID1_NODES, w1, b1);
	vector_relu(z1[0],a1,NUM_OF_HID1_NODES);
	linear_forward_nn(a1, NUM_OF_HID1_NODES, z2[0], NUM_OF_OUT_NODES, w2, b2);
	vector_sigmoid(z2[0],yhat[0], NUM_OF_OUT_NODES);

	double cost = compute_cost(1, NUM_OF_OUT_NODES, yhat, train_y);
  accuracy += checkPrediction( yhat, train_y);

}

void scale(double value, uint32_t MATRIX_ROW, uint32_t MATRIX_COL, double input_matrix[MATRIX_ROW][MATRIX_COL]) {
  for(int i = 0; i < MATRIX_ROW; ++i){
    for(int j = 0; j < MATRIX_COL; ++j){
      input_matrix[i][j] *= value;
    }
  }
}

double backward(double train_y[NUM_OF_FEATURES], double cum_dZ2[1][NUM_OF_HID1_NODES]){
	matrix_matrix_sub(NUM_OF_OUT_NODES, NUM_OF_OUT_NODES, yhat, train_y, dZ2);

  scale(1.0/50.0, 1, NUM_OF_OUT_NODES, dZ2);
  matrix_matrix_sum(1, NUM_OF_OUT_NODES, cum_dZ2, dZ2, cum_dZ2);


  //printf("dW2 before");
  //matrix_print(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, dW2);

  //linear_backward(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES,  1, dZ2, a1, dW2, db2);
  linear_backward(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES,  1, cum_dZ2, a1, dW2, db2);



	matrix_transpose(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, w2, W2_T);
	//matrix_matrix_multiplication(NUM_OF_HID1_NODES, NUM_OF_OUT_NODES, NUM_OF_OUT_NODES, W2_T, dZ2, dA1);
  matrix_matrix_multiplication(NUM_OF_HID1_NODES, NUM_OF_OUT_NODES, NUM_OF_OUT_NODES, W2_T, cum_dZ2, dA1);


	relu_backward(1, NUM_OF_HID1_NODES, dA1, z1, dZ1);
	linear_backward(NUM_OF_HID1_NODES, NUM_OF_FEATURES, 1, dZ1, train_x, dW1, db1);

  //printf("db2 after");
  //matrix_print(1, NUM_OF_OUT_NODES, db2);

  //printf("------------------------ \n");

}


void update(){

  weights_update(NUM_OF_HID1_NODES, NUM_OF_FEATURES, learning_rate, dW1, w1);
  weights_update(NUM_OF_HID1_NODES, NUM_OF_FEATURES, learning_rate, db1, b1);
  weights_update(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, learning_rate, dW2, w2);
  weights_update(NUM_OF_OUT_NODES, NUM_OF_HID1_NODES, learning_rate, db2, b2);
}


void iteration(int i, double raw_x[NUM_OF_FEATURES], double train_y[NUM_OF_FEATURES]){
  //double cum_dZ1[1][NUM_OF_HID1_NODES];
  double cum_dZ2[1][NUM_OF_OUT_NODES];

	//printf("it %d \n", i);
  forward(raw_x, train_y);
  //printf("--------- \n");
	backward(train_y, cum_dZ2);
}


int main(){
  init();

  for(int i = 0; i < EPOCH; ++i){
    accuracy = 0;
		for(int j = 0; j < CASES; ++j){
			iteration(i, raw_x[j], train_y[j]);
		}
    update();
    printf("(it %d) Train accuracy: %d \n", i, accuracy);
    printf("------------------------ \n");
	}
}
