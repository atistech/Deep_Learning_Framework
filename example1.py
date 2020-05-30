import framework.DigitClassifier as dc

#get instance
classifier = dc.DigitClassifier()

#print result
print(classifier.getResult("example1.png"))

#show result figure
classifier.showResultFigure("example1.png")
