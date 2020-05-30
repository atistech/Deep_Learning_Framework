import framework.ClothingClassifier as cc

#get instance
classifier = cc.ClothingClassifier()

#print result
print(classifier.getResult("example2.png"))

#show result figure
classifier.showResultFigure("example2.png")