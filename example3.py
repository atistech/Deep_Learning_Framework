import framework.ImageClassifier as ic

#get instance
classifier = ic.ImageClassifier()

#print result
print(classifier.getResult("example3.png"))

#show result figure
classifier.showResultFigure("example3.png")