using namespace std;

#include "./SLP.h"
#include "stdio.h"
#include <random>
#include <iostream>

SLP_Node::SLP_Node(unsigned length, float learningRate)
{
	this->weights = vector<float>(length, 0.0);
	this->learningRate = learningRate;
	default_random_engine generator;
	uniform_real_distribution<float> distribution(0.0, 1.0);
	for (unsigned weightIndex = 0; weightIndex < length; ++weightIndex)
	{
		this->weights[weightIndex] = distribution(generator);
	}	
}

void SLP_Node::calculateOutput(vector<unsigned char> inputVector)
{
	// TODO: Handle error for length mismatch.
	this->output = 0.0;
	for (unsigned index = 0; index < inputVector.size(); ++index)
	{
		this->output += inputVector[index] * this->weights[index];
	}
	this->output /= inputVector.size();
}

void SLP_Node::adjustWeights(vector<unsigned char> inputVector, float error)
{
	// TODO: Handle error for length mismatch.
	for (unsigned index = 0; index < inputVector.size(); ++index)
	{
		this->weights[index] += inputVector[index] * error * this->learningRate;
	}
}

float SLP_Node::getOutput()
{
	return this->output;
}

SLP::SLP(unsigned nodeCount, float learningRate, unsigned inputVectorSize)
{
	this->layer = vector<SLP_Node>();
	for (unsigned nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex)
	{
		this->layer.push_back(SLP_Node(inputVectorSize, learningRate));
	}
	this->learningRate = learningRate;
}

void SLP::train(vector<vector<unsigned char> > featureVectors, vector<unsigned char> labels)
{
	unsigned correct = 0;
	for (unsigned inputIndex = 0; inputIndex < featureVectors.size(); ++inputIndex)
	{
		vector<unsigned char> inputVector = featureVectors[inputIndex];
		unsigned inputLabel = (unsigned)labels[inputIndex];
		for (unsigned nodeIndex = 0; nodeIndex < this->layer.size(); ++nodeIndex)
		{
			float expectedOutput = 0;
			if (nodeIndex == inputLabel)
			{
				expectedOutput = 1.0;
			}
			this->layer[nodeIndex].calculateOutput(inputVector);
			float error = expectedOutput - this->layer[nodeIndex].getOutput();
			this->layer[nodeIndex].adjustWeights(inputVector, error);
		}
		if (this->getGuess() == inputLabel)
		{
			correct++;
		}
		printf("Training: %u/%u    Correct: %.2f%%\r", inputIndex + 1, (unsigned)featureVectors.size(), ((float)correct) / inputIndex * 100);
	}
	cout << endl;
}

void SLP::test(vector<vector<unsigned char> > featureVectors, vector<unsigned char> labels)
{
	unsigned correct = 0;
	for (unsigned inputIndex = 0; inputIndex < featureVectors.size(); ++inputIndex)
	{
		vector<unsigned char> inputVector = featureVectors[inputIndex];
		unsigned inputLabel = (unsigned)labels[inputIndex];
		for (unsigned nodeIndex = 0; nodeIndex < this->layer.size(); ++nodeIndex)
		{
			float expectedOutput = 0;
			if (nodeIndex == inputLabel)
			{
				expectedOutput = 1.0;
			}
			this->layer[nodeIndex].calculateOutput(inputVector);
		}
		if (this->getGuess() == inputLabel)
		{
			correct++;
		}
		printf("Testing: %u/%u     Correct: %.2f%%\r", inputIndex + 1, (unsigned)featureVectors.size(), ((float)correct) / inputIndex * 100);
	}
	cout << endl;
}

unsigned char SLP::getGuess()
{
	unsigned guess = 0;
	for (unsigned nodeIndex = 0; nodeIndex < this->layer.size(); ++nodeIndex)
	{
		if (this->layer[nodeIndex].getOutput() > this->layer[guess].getOutput())
		{
			guess = nodeIndex;
		}
	}
	return guess;
}