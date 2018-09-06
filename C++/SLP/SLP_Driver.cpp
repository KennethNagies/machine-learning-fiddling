using namespace std;
#include "../../MNIST/MNIST.h"
#include "./SLP.h"
#include <iostream>
#include <stdio.h>

int main()
{
	// Get the testing and training images.
	vector<MNIST_Image> trainingImages = MNIST_Image::getTrainingImages();
	vector<MNIST_Image> testingImages = MNIST_Image::getTestingImages();
	cout << "Simplifying Training Images:" << endl;
	for (unsigned imageIndex = 0; imageIndex < trainingImages.size(); ++imageIndex)
	{
		trainingImages[imageIndex].simplify();
		printf("%u/%u\r", imageIndex + 1, (unsigned)trainingImages.size());
	}
	cout << endl << "Simplifying Testing Images:" << endl;
	for (unsigned imageIndex = 0; imageIndex < testingImages.size(); ++imageIndex)
	{
		testingImages[imageIndex].simplify();
		printf("%u/%u\r", imageIndex + 1, (unsigned)testingImages.size());
	}
	cout << endl;

	// Get the feature vectors and labels from the images.
	vector<vector<unsigned char> > trainingFeatureVectors = vector<vector<unsigned char> >();
	vector<unsigned char> trainingLabels = vector<unsigned char>();
	for (vector<MNIST_Image>::iterator imPtr = trainingImages.begin(); imPtr != trainingImages.end(); ++imPtr)
	{
		trainingFeatureVectors.push_back((*imPtr).getImageVector());
		trainingLabels.push_back((*imPtr).getLabel());
	}
	vector<vector<unsigned char> > testingFeatureVectors = vector<vector<unsigned char> >();
	vector<unsigned char> testingLabels = vector<unsigned char>();
	for (vector<MNIST_Image>::iterator imPtr = testingImages.begin(); imPtr != testingImages.end(); ++imPtr)
	{
		testingFeatureVectors.push_back((*imPtr).getImageVector());
		testingLabels.push_back((*imPtr).getLabel());
	}

	// Train and test the SLP.
	SLP slp = SLP(10, 0.07, 28*28);
	slp.train(trainingFeatureVectors, trainingLabels);
	slp.test(testingFeatureVectors, testingLabels);
}