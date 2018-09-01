#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "../../MNIST/MNIST.h"

int main(int argc, const char* argv)
{
	MNIST_Image* trainingImages = GetTrainingImages();
	MNIST_Image* testingImages = GetTestingImages();
	printf("Simplifying Training Images:\n");
	for (uint32_t imageIndex = 0; imageIndex < 60000; ++imageIndex)
	{
		SimplifyImage(&(*(trainingImages + imageIndex)));
		printf("%d/%d\r", imageIndex + 1, 60000);
	}
	printf("\n");
	for (uint32_t imageIndex = 0; imageIndex < 10000; ++imageIndex)
	{
		SimplifyImage(&(*(testingImages + imageIndex)));
		printf("%d/%d\r", imageIndex + 1, 10000);
	}
	printf("\n");
	printImage(*trainingImages);
	printImage(*testingImages);
	FreeImages(trainingImages, 60000);
	FreeImages(testingImages, 10000);
}