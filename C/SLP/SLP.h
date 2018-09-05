#ifndef SLP_H
#define SLP_H

#include <inttypes.h>
#include "../../MNIST/MNIST.h"

typedef struct _SLP_Node
{
	float output;
	uint32_t width;
	uint32_t height;
	float** weights;
} SLP_Node;

typedef struct _SLP
{
	// The node layer.
	SLP_Node* layer;
	// The number of nodes in the layer.
	uint32_t size;
	// The learning rate for the SLP.
	float learningRate;
} SLP;

SLP InitSLP(uint32_t size, float learningRate);

void FreeSLP(SLP slp);

void Train(SLP* slp, MNIST_Image* trainingImages, uint32_t imageCount);

void Test(SLP* slp, MNIST_Image* testingImages, uint32_t imageCount);

#endif