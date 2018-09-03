#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include "./MNIST.h"

MNIST_Image InitImage(uint8_t* imageVector, uint8_t label, uint32_t length)
{
	MNIST_Image newImage;
	newImage.imageVector = malloc(length * sizeof(uint8_t));
	for (uint32_t byteIndex = 0; byteIndex < length; ++byteIndex)
	{
		*(newImage.imageVector + byteIndex) = *(imageVector + byteIndex);
	}
	newImage.label = label;
	newImage.length = length;
	return newImage;
}

void SimplifyImage(MNIST_Image* imagePtr)
{
	for (uint32_t byteIndex = 0; byteIndex < imagePtr->length; ++byteIndex)
	{
		if (*(imagePtr->imageVector + byteIndex) > 0)
		{
			*(imagePtr->imageVector + byteIndex) = 1;
		}
	}
}

void FreeImage(MNIST_Image image)
{
	free(image.imageVector);
}

void FreeImages(MNIST_Image* images, uint32_t length)
{
	for (uint32_t i = 0; i < length; i++)
	{
		FreeImage(*(images + i));
	}
	free(images);
}

uint32_t EndianConvert(uint32_t input)
{
	uint32_t b0 = (input & 0x000000ff) << 24u;
	uint32_t b1 = (input & 0x0000ff00) << 8u;
	uint32_t b2 = (input & 0x00ff0000) >> 8u;
	uint32_t b3 = (input & 0xff000000) >> 24u;
	return b0 | b1 | b2 | b3;
}

MNIST_Image* GetImages(char* imageFileName, char* labelFileName)
{
	FILE* imageFilePointer;
	FILE* labelFilePointer;
	imageFilePointer = fopen(imageFileName, "rb");
	labelFilePointer = fopen(labelFileName, "rb");
	// Skip the unnessecary info
	fseek(imageFilePointer, 4, SEEK_SET);
	fseek(labelFilePointer, 8, SEEK_SET);
	// Get the number of images.
	uint32_t imageCount;
	fread(&imageCount, 4, 1, imageFilePointer);
	imageCount = EndianConvert(imageCount);
	// Get the number of rows in an image.
	uint32_t rows;
	fread(&rows, 4, 1, imageFilePointer);
	// Get the number of columns in an image.
	uint32_t cols;
	fread(&cols, 4, 1, imageFilePointer);
	uint32_t endianCheck = 0x000000ff;
	char* eChar = ((char*)&endianCheck);
	bool isBigEndian = eChar == 0x00;
	if (!isBigEndian)
	{
		rows = EndianConvert(rows);
		cols = EndianConvert(cols);
	}
	// Allocate memory for the image and label buffers.
	uint8_t* imageBuffer = malloc(rows * cols * imageCount * sizeof(uint8_t));
	uint8_t* labelBuffer = malloc(imageCount * sizeof(uint8_t));
	// Read in all the images.
	fread(imageBuffer, 1, rows*cols*imageCount, imageFilePointer);
	fclose(imageFilePointer);
	// Read in all the labels.
	fread(labelBuffer, 1, imageCount, labelFilePointer);
	fclose(labelFilePointer);

	MNIST_Image* images = malloc(imageCount * sizeof(MNIST_Image));

	uint8_t* tempImageVector = malloc(rows * cols * sizeof(uint8_t));
	for (uint32_t imageIndex = 0; imageIndex < imageCount; ++imageIndex)
	{
		for (uint32_t byteIndex = 0; byteIndex < rows * cols; ++byteIndex)
		{
			*(tempImageVector + byteIndex) = *(imageBuffer + (imageIndex * (rows * cols)) + byteIndex);
		}
		*(images + imageIndex) = InitImage(tempImageVector, *(labelBuffer + imageIndex), rows * cols);
		printf("%d/%d\r", imageIndex + 1, imageCount);
	}
	printf("\n");
	free(tempImageVector);
	free(imageBuffer);
	free(labelBuffer);
	return images;
}

MNIST_Image* GetTrainingImages()
{
	printf("Reading Training Images:\n");
	MNIST_Image* images = GetImages("./MNIST/TrainingImages", "./MNIST/TrainingLabels");
	return images;
}

MNIST_Image* GetTestingImages()
{
	printf("Reading Testing Images:\n");
	MNIST_Image* images = GetImages("./MNIST/TestingImages", "./MNIST/TestingLabels");
	return images;
}

void printImage(MNIST_Image image)
{
	printf("%d\n", image.label);
	char* rowString = malloc(31 * sizeof(char));
	*(rowString + 30) = '\0';
	for (uint8_t i = 0; i < 30; ++i)
	{
		*(rowString + i) = '-';
	}
	printf("%s\n", rowString);
	*(rowString) = '|';
	*(rowString + 29) = '|';
	for (uint32_t rowIndex = 0; rowIndex < 28; ++rowIndex)
	{
		for (uint32_t colIndex = 0; colIndex < 28; ++ colIndex)
		{
			if (*(image.imageVector + (rowIndex * 28) + colIndex) == 0)
			{
				*(rowString + colIndex + 1) = ' ';
			}
			else
			{
				*(rowString + colIndex + 1) = '#';
			}
		}
		printf("%s\n", rowString);
	}
	for (uint8_t i = 0; i < 30; ++i)
	{
		*(rowString + i) = '-';
	}
	printf("%s\n", rowString);
	free(rowString);
}