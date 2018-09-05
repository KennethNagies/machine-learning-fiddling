#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include "./MNIST.h"

/*
 * Initialize an MNIST_Image with the given image and label.
 * @param imageVector: The image vector to clone into the MNIST_Image.
 * @param label: The label for the MNIST_Image.
 * @param width: The width of the image.
 * @param height: The height of the image.
 * @return: The initialized MNIST_Image.
 */
MNIST_Image InitImage(uint8_t* imageVector, uint8_t label, uint32_t width, uint32_t height)
{
	MNIST_Image newImage;
	newImage.image = malloc(height * sizeof(uint32_t*));
	uint32_t byteIndex = 0;
	for (uint32_t rowIndex = 0; rowIndex < height; ++rowIndex)
	{
		*(newImage.image + rowIndex) = malloc(width * sizeof(uint8_t));
		for (uint32_t colIndex = 0; colIndex < width; ++colIndex)
		{
			*(*(newImage.image + rowIndex) + colIndex) = *(imageVector + byteIndex);
			byteIndex++;
		}
	}
	newImage.label = label;
	newImage.width = width;
	newImage.height = height;
	return newImage;
}

/*
 * Reduce all values in the given image to ones and zeroes.
 * @param imagePtr: A pointer to the MNIST_Image to simplify.
 */
void SimplifyImage(MNIST_Image* imagePtr)
{
	for (uint32_t rowIndex = 0; rowIndex < imagePtr->height; ++rowIndex)
	{
		for (uint32_t colIndex = 0; colIndex < imagePtr->width; ++colIndex)
		{
			if (*(*(imagePtr->image + rowIndex) + colIndex) > 0)
			{
				*(*(imagePtr->image + rowIndex) + colIndex) = 1;
			}
		}
	}
}

/*
 * Free an MNIST_Image.
 * @param image: The image to free.
 */
void FreeImage(MNIST_Image image)
{
	for (uint32_t rowIndex = 0; rowIndex < image.height; ++rowIndex)
	{
		free(*(image.image + rowIndex));
	}
	free(image.image);
}

/*
 * Free an array of MNIST_Images.
 * @param images: The array of MNIST_Images.
 * @param length: The number of images.
 */
void FreeImages(MNIST_Image* images, uint32_t length)
{
	for (uint32_t i = 0; i < length; i++)
	{
		FreeImage(*(images + i));
	}
	free(images);
}

/*
 * Convert a given 32 bit integer from little to big endian or vice versa.
 * @param input: The number to convert.
 * @return: The number with its endian type switched.
 */
uint32_t EndianConvert(uint32_t input)
{
	uint32_t b0 = (input & 0x000000ff) << 24u;
	uint32_t b1 = (input & 0x0000ff00) << 8u;
	uint32_t b2 = (input & 0x00ff0000) >> 8u;
	uint32_t b3 = (input & 0xff000000) >> 24u;
	return b0 | b1 | b2 | b3;
}

/*
 * Reads in images and labels from the proved files and returns an array of MNIST_Image structs.
 * @param imageFileName: string containing the name and relative path of the image file.
 * @param labelFileName: string containing the name and relative path of the label file.
 * @return: An array of MNIST_Image structs.
 */
MNIST_Image* GetImages(char* imageFileName, char* labelFileName)
{
	FILE* imageFilePointer;
	FILE* labelFilePointer;
	imageFilePointer = fopen(imageFileName, "rb");
	labelFilePointer = fopen(labelFileName, "rb");
	// Skip the unnessecary info.
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

	// Handle endian conversion if necessary.
	uint32_t endianCheck = 0x000000ff;
	char* eChar = ((char*)&endianCheck);
	bool isBigEndian = *eChar == 0x00;
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

	// Construct the image array.
	MNIST_Image* images = malloc(imageCount * sizeof(MNIST_Image));
	uint8_t* tempImageVector = malloc(rows * cols * sizeof(uint8_t));
	for (uint32_t imageIndex = 0; imageIndex < imageCount; ++imageIndex)
	{
		for (uint32_t byteIndex = 0; byteIndex < rows * cols; ++byteIndex)
		{
			*(tempImageVector + byteIndex) = *(imageBuffer + (imageIndex * (rows * cols)) + byteIndex);
		}
		*(images + imageIndex) = InitImage(tempImageVector, *(labelBuffer + imageIndex), rows, cols);
		printf("%d/%d\r", imageIndex + 1, imageCount);
	}
	printf("\n");
	free(tempImageVector);
	free(imageBuffer);
	free(labelBuffer);
	return images;
}

/*
 * Get all the training MNIST_Image structs.
 * @return: An array of the training images. 
 */
MNIST_Image* GetTrainingImages()
{
	printf("Reading Training Images:\n");
	MNIST_Image* images = GetImages("./MNIST/TrainingImages", "./MNIST/TrainingLabels");
	return images;
}

/*
 * Get all the testing MNIST_Image structs.
 * @return: An array of the testing images.
 */
MNIST_Image* GetTestingImages()
{
	printf("Reading Testing Images:\n");
	MNIST_Image* images = GetImages("./MNIST/TestingImages", "./MNIST/TestingLabels");
	return images;
}

/*
 * Print the given MNIST_Image struct to the console.
 */
void printImage(MNIST_Image image)
{
	printf("%d\n", image.label);
	char* rowString = malloc((image.width + 3) * sizeof(char));
	*(rowString + (image.width + 2)) = '\0';
	for (uint8_t i = 0; i < image.width + 2; ++i)
	{
		*(rowString + i) = '-';
	}
	printf("%s\n", rowString);
	*(rowString) = '|';
	*(rowString + image.width + 1) = '|';
	for (uint32_t rowIndex = 0; rowIndex < image.height; ++rowIndex)
	{
		for (uint32_t colIndex = 0; colIndex < image.width; ++colIndex)
		{
			if (*(*(image.image + rowIndex) + colIndex) == 0)
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
	for (uint8_t i = 0; i < image.width + 2; ++i)
	{
		*(rowString + i) = '-';
	}
	printf("%s\n", rowString);
	free(rowString);
}