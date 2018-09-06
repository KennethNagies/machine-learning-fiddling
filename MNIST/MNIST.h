#ifndef MNIST_H
#define MNIST_H
#include <vector>
#include <string>

class MNIST_Image
{
	public:
		MNIST_Image(unsigned char* imageArray, unsigned char label, unsigned width, unsigned height);
		// Reduces the image to contain only ones and zeroes.
		void simplify();
		void printImage();
		std::vector<unsigned char> getImageVector();
		std::vector<std::vector<unsigned char> > getImage2D();
		unsigned getWidth();
		unsigned getHeight();
		unsigned char getLabel();
		static std::vector<MNIST_Image> getTrainingImages();
		static std::vector<MNIST_Image> getTestingImages();

	private:
		std::vector<unsigned char> imageVector;
		unsigned width;
		unsigned height;
		unsigned char label;
		static unsigned EndianConvert(unsigned input);
		static std::vector<MNIST_Image> GetImages(std::string imageFileName, std::string labelFileName);
};
#endif