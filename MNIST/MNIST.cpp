using namespace std;
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "./MNIST.h"


MNIST_Image::MNIST_Image(unsigned char* imageArray, unsigned char label, unsigned width, unsigned height)
{
	this->imageVector = vector<unsigned char>(width * height, 0);
	for (unsigned i = 0; i < width * height; ++i)
	{
		this->imageVector[i] = imageArray[i];
	}
	this->label = label;
	this->width = width;
	this->height = height;
}

void MNIST_Image::simplify()
{
	for (unsigned index = 0; index < this->imageVector.size(); ++index)
	{
		if (this->imageVector[index] > (unsigned char)0)
		{
			this->imageVector[index] = (unsigned char)1;
		}
	}
}

void MNIST_Image::printImage()
{
	cout << (unsigned)this->label << endl;
	vector<vector<unsigned char> > image2D = this->getImage2D();
	string outString = string(this->width + 2, '-');
	cout << outString << endl;
	outString[0] = '|';
	outString[this->width + 1] = '|';
	for (unsigned rowIndex = 0; rowIndex < this->height; ++ rowIndex)
	{
		vector<unsigned char> row = image2D[rowIndex];
		for (unsigned byteIndex = 0; byteIndex < row.size(); ++byteIndex)
		{
			if (row[byteIndex] > (unsigned char)0)
			{
				outString[byteIndex + 1] = '#';
			}
			else
			{
				outString[byteIndex + 1] = ' ';
			}
		}
		cout << outString << endl;
	}
	for (unsigned i = 0; i < this->width + 2; ++i)
	{
		outString[i] = '-';
	}
	cout << outString << endl;
}

vector<unsigned char> MNIST_Image::getImageVector()
{
	vector<unsigned char> copy (this->imageVector);
	return copy;
}

vector<vector<unsigned char> > MNIST_Image::getImage2D()
{
	vector<vector<unsigned char> > image2D = vector<vector<unsigned char> >();
	for (unsigned rowIndex = 0; rowIndex < this->height; ++rowIndex)
	{
		image2D.push_back(vector<unsigned char>(this->width, (unsigned char)0));
		for (unsigned colIndex = 0; colIndex < this->width; ++colIndex)
		{
			image2D[rowIndex][colIndex] = this->imageVector[(rowIndex * this->width) + colIndex];
		}
	}
	return image2D;
}

unsigned MNIST_Image::getWidth()
{
	return this->width;
}

unsigned MNIST_Image::getHeight()
{
	return this->height;
}

unsigned char MNIST_Image::getLabel()
{
	return this->label;
}

/*
 * Get all the training MNIST_Image structs.
 * @return: An array of the training images. 
 */
vector<MNIST_Image> MNIST_Image::getTrainingImages()
{
	cout << "Reading Training Images:" << endl;
	vector<MNIST_Image> images = MNIST_Image::GetImages("./MNIST/TrainingImages", "./MNIST/TrainingLabels");
	return images;
}

/*
 * Get all the testing MNIST_Image structs.
 * @return: An array of the testing images.
 */
vector<MNIST_Image> MNIST_Image::getTestingImages()
{
	cout << "Reading Testing Images:" << endl;
	vector<MNIST_Image> images = MNIST_Image::GetImages("./MNIST/TestingImages", "./MNIST/TestingLabels");
	return images;
}

/*
 * Convert a given 32 bit integer from little to big endian or vice versa.
 * @param input: The number to convert.
 * @return: The number with its endian type switched.
 */
unsigned MNIST_Image::EndianConvert(unsigned input)
{
	unsigned b0 = (input & 0x000000ff) << 24u;
	unsigned b1 = (input & 0x0000ff00) << 8u;
	unsigned b2 = (input & 0x00ff0000) >> 8u;
	unsigned b3 = (input & 0xff000000) >> 24u;
	return b0 | b1 | b2 | b3;
}

vector<MNIST_Image> MNIST_Image::GetImages(string imageFileName, string labelFileName)
{
	FILE* imageFilePointer;
	FILE* labelFilePointer;
	imageFilePointer = fopen(imageFileName.c_str(), "rb");
	labelFilePointer = fopen(labelFileName.c_str(), "rb");
	// Skip the unnessecary info.
	fseek(imageFilePointer, 4, SEEK_SET);
	fseek(labelFilePointer, 8, SEEK_SET);
	// Get the number of images.
	unsigned imageCount;
	fread(&imageCount, 4, 1, imageFilePointer);
	imageCount = EndianConvert(imageCount);
	// Get the number of rows in an image.
	unsigned rows;
	fread(&rows, 4, 1, imageFilePointer);
	// Get the number of columns in an image.
	unsigned cols;
	fread(&cols, 4, 1, imageFilePointer);

	// Handle endian conversion if necessary.
	unsigned endianCheck = 0x000000ff;
	char* eChar = ((char*)&endianCheck);
	bool isBigEndian = *eChar == 0x00;
	if (!isBigEndian)
	{
		rows = MNIST_Image::EndianConvert(rows);
		cols = MNIST_Image::EndianConvert(cols);
	}

	// Allocate memory for the image and label buffers.
	unsigned char* imageBuffer = (unsigned char*)malloc(rows * cols * imageCount * sizeof(unsigned char));
	unsigned char* labelBuffer = (unsigned char*)malloc(imageCount * sizeof(unsigned char));

	// Read in all the images.
	fread(imageBuffer, 1, rows*cols*imageCount, imageFilePointer);
	fclose(imageFilePointer);

	// Read in all the labels.
	fread(labelBuffer, 1, imageCount, labelFilePointer);
	fclose(labelFilePointer);

	// Construct the image array.
	vector<MNIST_Image> images = vector<MNIST_Image>();
	unsigned char* tempImageVector = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));
	for (unsigned imageIndex = 0; imageIndex < imageCount; ++imageIndex)
	{
		for (unsigned byteIndex = 0; byteIndex < rows * cols; ++byteIndex)
		{
			*(tempImageVector + byteIndex) = *(imageBuffer + (imageIndex * (rows * cols)) + byteIndex);
		}
		images.push_back(MNIST_Image(tempImageVector, *(labelBuffer + imageIndex), rows, cols));
		printf("%d/%d\r", imageIndex + 1, imageCount);
	}
	printf("\n");
	free(tempImageVector);
	free(imageBuffer);
	free(labelBuffer);
	return images;
}
