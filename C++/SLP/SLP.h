#ifndef SLP_H
#define SLP_H

#include <vector>

class SLP_Node
{
	public:
		SLP_Node(unsigned length, float learningRate);
		float getOutput();
		void calculateOutput(std::vector<unsigned char> inputVector);
		void adjustWeights(std::vector<unsigned char> inputVector, float error);

	private:
		float output;
		std::vector<float> weights;
		float learningRate;
};

class SLP
{
	public:
		SLP(unsigned nodeCount, float learningRate, unsigned inputVectorSize);
		void train(vector<vector<unsigned char> > featureVectors, vector<unsigned char> labels);
		void test(vector<vector<unsigned char> > featureVectors, vector<unsigned char> labels);
	private:
		std::vector<SLP_Node> layer;
		float learningRate;
		unsigned char getGuess();
};

#endif