#ifndef MNIST_H
#define MNIST_H

#include <inttypes.h>

typedef struct __MNIST_Image
{
	uint8_t* imageVector;
	uint32_t length;
	uint8_t label;
} MNIST_Image;

MNIST_Image InitImage(uint8_t* imageVector, uint8_t label, uint32_t length);

void SimplifyImage(MNIST_Image* imagePtr);

void FreeImages(MNIST_Image* images, uint32_t length);

MNIST_Image* GetTrainingImages();

MNIST_Image* GetTestingImages();

void printImage(MNIST_Image image);

#endif