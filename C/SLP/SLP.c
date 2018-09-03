#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "./SLP.h"

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

void GetOutput(SLP_Node* node, uint8_t* inputVector)
{
	node->output = 0.0;
	for (uint32_t byteIndex = 0; byteIndex < node->length; ++byteIndex)
	{
		node->output += ((float)(*(inputVector + byteIndex))) * ((float)(*(node->weights + byteIndex)));
	}
	//printf("%.2f\n", node.output);
	node->output /= ((float)(28*28));
	//printf("%0.2f\n", node->output);
}

void AdjustWeights(SLP_Node* node, uint8_t* inputVector, float error, float learningRate)
{
	for (uint32_t weightIndex = 0; weightIndex < node->length; ++weightIndex)
	{
		*(node->weights + weightIndex) += learningRate * error * ((float)(*(inputVector + weightIndex)));
	}
}

void FreeNode(SLP_Node node)
{
	free(node.weights);
}

void FreeNodes(SLP_Node* nodes, uint32_t nodeCount)
{
	for (uint32_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex)
	{
		FreeNode(*(nodes + nodeIndex));
	}
	free(nodes);
}

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

void FreeSLP(SLP slp)
{
	for (uint32_t nodeIndex = 0; nodeIndex < slp.size; ++nodeIndex)
	{
		FreeNode(*(slp.layer + nodeIndex));
	}
	free(slp.layer);
}

uint8_t GetGuess(SLP* slp)
{
	uint8_t guess = 0;
	for (uint8_t nodeIndex = 0; nodeIndex < slp->size; ++nodeIndex)
	{
		//printf("%.2f, %.2f\n", (*(slp->layer + nodeIndex)).output, (*(slp->layer + guess)).output);
		if ((*(slp->layer + nodeIndex)).output > (*(slp->layer + guess)).output)
		{
			guess = nodeIndex;
		}
	}
	return guess;
}

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
		//printf("%d\n", GetGuess(slp));
		if (GetGuess(slp) == image.label)
		{
			correct++;
		}
		printf("Training: %d/%d    Correct: %.2f%%\r", imageIndex + 1, imageCount, ((float)(correct))/((float)(imageIndex + 1)) * 100);
	}
	printf("\n");
	free(expectedOuts);
}

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