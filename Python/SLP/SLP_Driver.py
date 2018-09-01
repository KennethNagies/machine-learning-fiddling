import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../MNIST"))
from MNIST import MNIST_Image
from SLP import SLP

def main():
	trainingImages = MNIST_Image.getTrainingData()
	testingImages = MNIST_Image.getTestingData()
	for image in trainingImages:
		image.simplify()
	for image in testingImages:
		image.simplify()
	slp = SLP(10, 28*28, 0.07)
	slp.train(trainingImages)
	slp.test(testingImages)



if __name__=="__main__":
	main()