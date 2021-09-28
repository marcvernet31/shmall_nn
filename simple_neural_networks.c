#include "simple_neural_networks.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// TODO
double single_in_single_out_nn(double  input, double weight) {
   	return input * weight;
}

// TODO
double weighted_sum(double * input, double * weight, uint32_t INPUT_LEN) {
	double output = 0;
	for(int i = 0; i < INPUT_LEN; ++i) output += input[i] * weight[i];

 return output;
}

// TODO
double multiple_inputs_single_output_nn(double * input, double *weight, uint32_t INPUT_LEN) {
	return weighted_sum(input, weight, INPUT_LEN);
}


void elementwise_multiple( double input_scalar, double *weight_vector, double *output_vector, double VECTOR_LEN) {
	// TODO: Use for loop to calculate output_vector
	for(int i = 0; i < VECTOR_LEN; ++i){
		output_vector[i] = input_scalar * weight_vector[i];
	}
}



void single_input_multiple_output_nn(double input_scalar, double *weight_vector, double *output_vector, double VECTOR_LEN){
  elementwise_multiple(input_scalar, weight_vector,output_vector,VECTOR_LEN);
}

// TODO
void matrix_vector_multiplication(double * input_vector, uint32_t INPUT_LEN, double * output_vector,
		uint32_t OUTPUT_LEN, double weights_matrix[OUTPUT_LEN][INPUT_LEN]) {
	for(int i = 0; i < OUTPUT_LEN; ++i){
		output_vector[i] = 0;
		for(int j = 0; j < INPUT_LEN; ++j){
			output_vector[i] += input_vector[j] * weights_matrix[i][j];
		}
	}
}


void multiple_inputs_multiple_outputs_nn(double * input_vector, uint32_t INPUT_LEN, double * output_vector,
		uint32_t OUTPUT_LEN, double weights_matrix[OUTPUT_LEN][INPUT_LEN]) {
	matrix_vector_multiplication(input_vector,INPUT_LEN,output_vector,OUTPUT_LEN,weights_matrix);
}


void hidden_nn( double *input_vector, uint32_t INPUT_LEN,
				uint32_t HIDDEN_LEN, double in_to_hid_weights[HIDDEN_LEN][INPUT_LEN],
				uint32_t OUTPUT_LEN, double hid_to_out_weights[OUTPUT_LEN][HIDDEN_LEN], double *output_vector) {
	/* TODO: Use matrix_vector_multiplication to calculate values for hidden_layer. Make sure that when you initialize
	   hidden_pred_vector variable then zero its value with for loop */
	double hidden_pred_vector[HIDDEN_LEN];
	matrix_vector_multiplication(input_vector, INPUT_LEN, hidden_pred_vector, HIDDEN_LEN, in_to_hid_weights);

	// TODO: Use matrix_vector_multiplication to calculate output layer values from hidden layer
	matrix_vector_multiplication(hidden_pred_vector, HIDDEN_LEN, output_vector, OUTPUT_LEN, hid_to_out_weights);

}


double find_error(double yhat, double y) {
	// TODO: Use math.h functions to calculate the error with double precision
	double base = yhat - y;
	double result = pow(base, 2 );
	return result;
}



void brute_force_learning( double input, double weight, double expected_value, double step_amount, uint32_t itr) {
   double prediction,error;
   double up_prediction, down_prediction, up_error, down_error;
   int i;
	 for(i=0;i<itr;i++){

		 prediction  = input * weight;
		 // TODO: Calculate the error

		 error = find_error(prediction, expected_value);

		 printf("Step: %d   Error: %f    Prediction: %f    Weight: %f\n", i, error, prediction, weight);

		 up_prediction =  input * (weight + step_amount);
		 up_error      =   powf((up_prediction - expected_value),2);

		 // TODO: Calculate down_prediction and down_error on the same way as up_prediction and up_error
		 down_prediction =  input * (weight - step_amount);
		 down_error      =  powf((down_prediction - expected_value),2);;

		 if(down_error <  up_error)
			 // TODO: Change weight value accordingly if down_error is smaller than up_error
			   weight  = weight - step_amount;
		 if(down_error >  up_error)
			 // TODO: Change weight value accordingly if down_error is larger than up_error
			   weight = weight + step_amount;
	 }
}



void linear_forward_nn(double *input_vector, uint32_t INPUT_LEN,
						double *output_vector, uint32_t OUTPUT_LEN,
						double weights_matrix[OUTPUT_LEN][INPUT_LEN], double *weights_b) {

	matrix_vector_multiplication(input_vector,INPUT_LEN, output_vector,OUTPUT_LEN,weights_matrix);

	for(int k=0;k<OUTPUT_LEN;k++){
		output_vector[k]+=weights_b[k];
	}

}

// TODO
double relu(double x){
	if(x > 0) return x;
	else return 0;
}


void vector_relu(double *input_vector, double *output_vector, uint32_t LEN) {
	  for(int i =0;i<LEN;i++){
		  output_vector[i] =  relu(input_vector[i]);
		}
}

// TODO:
// 1 / (1 + e^-x)
double sigmoid(double x) {
	 double result = 1 / (1 + exp(-x));
	 return result;
}


void vector_sigmoid(double * input_vector, double * output_vector, uint32_t LEN) {
	for (int i = 0; i < LEN; i++) {
		output_vector[i] = sigmoid(input_vector[i]);
	}
}


double compute_cost(uint32_t m, double yhat[m][1], double y[m][1]) {
	double cost = 0;
	// TODO: Calculate cost based on mathematical cost function formula
	double sum = 0;
	for(int i = 0; i < m; ++i){
		sum += y[i][0] * log(yhat[i][0]) + (1.0 - y[i][0]) * log(1.0 - yhat[i][0]);
	}
	cost = sum * (-1.0 / m);
	return cost;
}


void normalize_data_2d(uint32_t ROW, uint32_t COL, double input_matrix[ROW][COL], double output_matrix[ROW][COL]){
	double max =  -99999999;
	for(int i =0;i<ROW;i++){
	  for(int j =0;j<COL;j++){
		  if(input_matrix[i][j] >max){
			  max = input_matrix[i][j];
			}
		}
	}

	for(int i=0;i<ROW;i++){
		for(int j=0;j<COL;j++){
	    output_matrix[i][j] =  input_matrix[i][j]/max;
		}
	}
}


// Use this function to print matrix values for debugging
void matrix_print(uint32_t ROW, uint32_t COL, double A[ROW][COL]) {
	for(int i=0; i<ROW; i++){
			for(int j=0; j<COL; j++){
				printf(" %f ", A[i][j]);
			}
			printf("\n");
	}
	printf("\n\r");
}


void weights_random_initialization(uint32_t HIDDEN_LEN, uint32_t INPUT_LEN, double weight_matrix[HIDDEN_LEN][INPUT_LEN]) {
	double d_rand;

	/*Seed random number generator*/
	srand(1);

	for (int i = 0; i < HIDDEN_LEN; i++) {
		for (int j = 0; j < INPUT_LEN; j++) {
			/*Generate random numbers between 0 and 1*/
			d_rand = (rand() % 10);
			d_rand /= 10;
			weight_matrix[i][j] = d_rand;
		}
	}
}


void weightsB_zero_initialization(double * weightsB, uint32_t LEN){
	memset(weightsB, 0, LEN*sizeof(weightsB[0]));
}


// que fa la m ??
void relu_backward(uint32_t m, uint32_t LAYER_LEN, double dA[m][LAYER_LEN], double Z[m][LAYER_LEN], double dZ[m][LAYER_LEN]) {
	//TODO: implement derivative of relu function  You can can choose either to calculate for all example at the same time
	//or make iteratively. Check formula for derivative lecture 5 on slide 24
		for(int i = 0; i < LAYER_LEN; ++i){
			if(Z[0][i] >= 0){
				dZ[0][i] = dA[0][i]; // relu der. is 1
			}
			else{
				dZ[0][i] = 0; // relu der. is 1
			}
		}
}


// perque m ?
void linear_backward(uint32_t LAYER_LEN, uint32_t PREV_LAYER_LEN, uint32_t m, double dZ[m][LAYER_LEN],
		double A_prev[m][PREV_LAYER_LEN], double dW[LAYER_LEN][PREV_LAYER_LEN], double * db ){
	// TODO: implement linear backward. You can can choose either to calculate for all example at the same time (dw= 1/m *A_prev[T]*dZ;)
	//or make iteratively  (dw_iter= A_prev[T]*dZ;)
  matrix_matrix_multiplication(LAYER_LEN, m, PREV_LAYER_LEN, dZ, A_prev, dW);
  matrix_matrix_sum(LAYER_LEN, m, db, dZ, db); // funciona??

  /*
	for(int i = 0; i < LAYER_LEN; ++i){
		for(int j = 0; j < PREV_LAYER_LEN; ++j){
			dW[i][j] = (1.0 / m) * A_prev[0][j] * dZ[0][i];
		}
		db[i] += dZ[0][i];
	}
  */

}


void matrix_matrix_sum(uint32_t MATRIX_ROW, uint32_t MATRIX_COL,
									double input_matrix1[MATRIX_ROW][MATRIX_COL],
									double input_matrix2[MATRIX_COL][MATRIX_COL],
									double output_matrix[MATRIX_ROW][MATRIX_COL]) {
	for (int c = 0; c < MATRIX_ROW; c++) {
	      for (int d = 0; d < MATRIX_COL; d++) {
	        output_matrix[c][d] = input_matrix1[c][d]+input_matrix2[c][d];
	      }
	 }
}


void matrix_divide_scalar(uint32_t MATRIX_ROW, uint32_t MATRIX_COL, double scalar,
									double input_matrix[MATRIX_ROW][MATRIX_COL],
									double output_matrix[MATRIX_ROW][MATRIX_COL]) {
	for (int c = 0; c < MATRIX_ROW; c++) {
	      for (int d = 0; d < MATRIX_COL; d++) {
	        output_matrix[c][d] = input_matrix[c][d]/scalar;
	      }
	 }
}


void matrix_matrix_multiplication(uint32_t MATRIX1_ROW, uint32_t MATRIX1_COL, uint32_t MATRIX2_COL,
									double input_matrix1[MATRIX1_ROW][MATRIX1_COL],
									double input_matrix2[MATRIX1_COL][MATRIX2_COL],
									double output_matrix[MATRIX1_ROW][MATRIX2_COL]) {

	for(int k=0;k<MATRIX1_ROW;k++){
		 memset(output_matrix[k], 0, MATRIX2_COL*sizeof(output_matrix[0][0]));
	}
	double sum=0;
	for (int c = 0; c < MATRIX1_ROW; c++) {
	      for (int d = 0; d < MATRIX2_COL; d++) {
	        for (int k = 0; k < MATRIX1_COL; k++) {
	          sum += input_matrix1[c][k]*input_matrix2[k][d];
	        }
	        output_matrix[c][d] = sum;
	        sum = 0;
	      }
	 }
}


void matrix_matrix_sub(uint32_t MATRIX_ROW, uint32_t MATRIX_COL,
									double input_matrix1[MATRIX_ROW][MATRIX_COL],
									double input_matrix2[MATRIX_ROW][MATRIX_COL],
									double output_matrix[MATRIX_ROW][MATRIX_COL]) {
	for (int c = 0; c < MATRIX_ROW; c++) {
	      for (int d = 0; d < MATRIX_COL; d++) {
	        output_matrix[c][d] = input_matrix1[c][d]-input_matrix2[c][d];
	      }
	 }
}

// W1 = W1 - learning_rate * dW1
void weights_update(uint32_t MATRIX_ROW, uint32_t MATRIX_COL, double learning_rate,
									double dW[MATRIX_ROW][MATRIX_COL],
									double W[MATRIX_ROW][MATRIX_COL]) {
	//TODO: implement weights_update function
	for(int i = 0; i < MATRIX_ROW; ++i){
		for(int j = 0; j < MATRIX_COL; ++j){
			W[i][j] = W[i][j] - learning_rate * dW[i][j];
		}
	}
}


void matrix_multiply_scalar(uint32_t MATRIX_ROW, uint32_t MATRIX_COL, double scalar,
									double input_matrix[MATRIX_ROW][MATRIX_COL],
									double output_matrix[MATRIX_ROW][MATRIX_COL]) {
	for (int c = 0; c < MATRIX_ROW; c++) {
	      for (int d = 0; d < MATRIX_COL; d++) {
	        output_matrix[c][d] = input_matrix[c][d]*scalar;
	      }
	 }
}


void matrix_transpose(uint32_t ROW, uint32_t COL, double A[ROW][COL], double A_T[COL][ROW]) {
	for(int i=0; i<ROW; i++){
		for(int j=0; j<COL; j++){
			A_T[j][i]=A[i][j];
		}
	}
}
