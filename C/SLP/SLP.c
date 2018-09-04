#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "./SLP.h"

/*
 * Initialize an SLP_Node with the given length.
 * @param length: The length of the input and weight vectors.
 * @return: The initialized SLP_Node.
 */
SLP_Node InitNode(uint32_t length)
{
	SLP_Node node;
	node.output = 0.0;
	node.length = length;
	node.weights = malloc(length * sizeof(float));

	// Initialize all of the weights to be a random number between 0 and 1.
	for (uint32_t weightIndex = 0; weightIndex < length; ++weightIndex)
	{
		*(node.weights + weightIndex) = ((float)rand()) / ((float)RAND_MAX);
	}
	return node;
}

/*
 * Calculate the output for the given node.
 * @param node: A pointer to the node to calculate output for.
 * @param inputVector: The vector to calculate the output based on.
 */
void GetOutput(SLP_Node* node, uint8_t* inputVector)
{
	node->output = 0.0;
	for (uint32_t byteIndex = 0; byteIndex < node->length; ++byteIndex)
	{
		node->output += ((float)(*(inputVector + byteIndex))) * ((float)(*(node->weights + byteIndex)));
	}
	node->output /= ((float)(28*28));
}

/*
 * Adjust the weights of the given node based on the input and a given error and learning rate.
 * @param node: A pointer to the node.
 * @param inputVector: The input vector to calculate based on.
 * @param error: The calculated error. (excpected - actual)
 * @param learningRate: The learning rate for the node.
 */
void AdjustWeights(SLP_Node* node, uint8_t* inputVector, float error, float learningRate)
{
	for (uint32_t weightIndex = 0; weightIndex < node->length; ++weightIndex)
	{
		*(node->weights + weightIndex) += learningRate * error * ((float)(*(inputVector + weightIndex)));
	}
}

/*
 * Free the data in the given node.
 * @param node: The node to free.
 */
void FreeNode(SLP_Node node)
{
	free(node.weights);
}

/*
 * Free the given array of nodes.
 * @param nodes: The array of nodes to free.
 * @param nodeCount: The number of nodes in the array.
 */
void FreeNodes(SLP_Node* nodes, uint32_t nodeCount)
{
	for (uint32_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex)
	{
		FreeNode(*(nodes + nodeIndex));
	}
	free(nodes);
}

/*
 * Create a SLP with the given size and learning rate.
 * @param size: The number of nodes in the SLP.
 * @param learningRate: The learning rate of the SLP.
 * @return: An initialized SLP. 
 */
SLP InitSLP(uint32_t size, float learningRate)
{
	SLP slp;
	slp.size = size;
	slp.learningRate = learningRate;
	slp.layer = malloc(size * sizeof(SLP_Node));
	for (uint32_t nodeIndex = 0; nodeIndex < size; ++nodeIndex)
	{
		*(slp.layer + nodeIndex) = InitNode(28*28);
	}
	return slp;
}

/*
 * Free the data in the given SLP.
 * @param slp: The SLP to free.
 */
void FreeSLP(SLP slp)
{
	for (uint32_t nodeIndex = 0; nodeIndex < slp.size; ++nodeIndex)
	{
		FreeNode(*(slp.layer + nodeIndex));
	}
	free(slp.layer);
}

/*
 * Get the digit that the given SLP guesses based on the previous input.
 * @param slp: A pointer to the slp.
 * @return: The guess.
 */
uint8_t GetGuess(SLP* slp)
{
	uint8_t guess = 0;
	for (uint8_t nodeIndex = 0; nodeIndex < slp->size; ++nodeIndex)
	{
		if ((*(slp->layer + nodeIndex)).output > (*(slp->layer + guess)).output)
		{
			guess = nodeIndex;
		}
	}
	return guess;
}

/*
 * Train the SLP based on an array of MNIST_images.
 * @param slp: A pointer to the slp.
 * @param trainingImages: The images to train with.
 * @param imageCount: The number of images in the training set.
 */
void Train(SLP* slp, MNIST_Image* trainingImages, uint32_t imageCount)
{
	uint8_t* expectedOuts = malloc(slp->size * sizeof(uint8_t));
	for (uint32_t i = 0; i < slp->size; ++i)
	{
		*(expectedOuts + i) = 0;
	}
	uint32_t correct = 0;
	for (uint32_t imageIndex = 0; imageIndex < imageCount; ++imageIndex)
	{
		MNIST_Image image = *(trainingImages + imageIndex);
		*(expectedOuts + image.label) = 1;
		for (uint32_t nodeIndex = 0; nodeIndex < slp->size; ++nodeIndex)
		{
			SLP_Node* node = &(*(slp->layer + nodeIndex));
			GetOutput(node, image.imageVector);
			float error = ((float)(*(expectedOuts + nodeIndex))) - ((float)(node->output));
			AdjustWeights(node, image.imageVector, error, slp->learningRate);
		}
		*(expectedOuts + image.label) = 0;
		if (GetGuess(slp) == image.label)
		{
			correct++;
		}
		printf("Training: %d/%d    Correct: %.2f%%\r", imageIndex + 1, imageCount, ((float)(correct))/((float)(imageIndex + 1)) * 100);
	}
	printf("\n");
	free(expectedOuts);
}

/*
 * Test the SLP based on an array of test MNIST_Images.
 * @param slp: A pointer to the slp.
 * @param testingImages: An array of images to test with.
 * @param imageCount: The number of images in the testing set.
 */
void Test(SLP* slp, MNIST_Image* testingImages, uint32_t imageCount)
{
	uint32_t correct = 0;
	for (uint32_t imageIndex = 0; imageIndex < imageCount; ++imageIndex)
	{
		MNIST_Image image = *(testingImages + imageIndex);
		for (uint32_t nodeIndex = 0; nodeIndex < slp->size; ++nodeIndex)
		{
			SLP_Node* node = &(*(slp->layer + nodeIndex));
			GetOutput(node, image.imageVector);
		}
		if (GetGuess(slp) == image.label)
		{
			correct++;
		}
		printf("Testing:  %d/%d    Correct: %.2f%%\r", imageIndex + 1, imageCount, ((float)(correct))/((float)(imageIndex + 1)) * 100);
	}
	printf("\n");
}