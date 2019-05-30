import numpy as np
import matplotlib.pyplot as plt

class BinaryClassificator:
    def __init__(self,porog=150):
        self.porog = porog
        self.N=1000
        self.footballers = np.random.normal(170, 15, self.N)
        self.basketballers = np.random.normal(190, 10, self.N)
        self.bins = np.arange(130,231)
    def showInputData(self):
        plt.hist(self.footballers, self.bins, alpha=0.5, label='x')
        plt.hist(self.basketballers, self.bins, alpha=0.5, label='y')
        plt.legend(['Футболисты', 'Баскетболисты'])
        plt.show()
    def showCurve(self,x,y,title="Curve"):
        plt.plot(x, y)
        plt.title(title)
        plt.show()

    def porogClassificator(self,porog=150):
        self.porog = porog
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(self.N):
            if (self.footballers[i] >= porog):
                classificator = 0
                predict = 1
                fp += 1
            else:
                classificator = 0
                predict = 0
                tn += 1
            if (self.basketballers[i] >= porog):
                classificator = 1
                predict = 1
                tp += 1
            else:
                classificator = 1
                predict = 0
                fn += 1
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            recall = tp / (tp + fn)
        try:
            precision = tp / (tp + fp)
            f1 = (1 + 1) * precision * recall / (precision + recall)
        except:
            precision=0
            f1=0
        return {"tp":tp,"tn":tn,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"accuracy":accuracy,"f1_score":f1,"first_error":fp,"second_error":fn}
    def getRandomClass(self,p):
        if (np.random.random()<= p):
            return 1
        else:
            return 0
    def probabilityClassificator(self,step=0.01,probability=0.5):
        self.probability = probability
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        #basketballs
        for i in range(self.N):
            randomClass = self.getRandomClass(probability)
            if (randomClass == 1 ):
                tp+=1
            else:
                fn+=1
            randomClass = self.getRandomClass(probability)
            if (randomClass == 1):
                fp += 1
            else:
                tn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        try:
            precision = tp / (tp + fp)
            f1 = (1 + 1) * precision * recall / (precision + recall)
        except:
            precision=0
            f1=0
        return {"tp":tp,"tn":tn,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"accuracy":accuracy,"f1_score":f1,"first_error":fp,"second_error":fn}
    def getSquare(self,x,y):
        s=0
        for i in range(len(x)-1):
            h = x[i+1]-x[i]
            ab = y[i+1]+y[i]
            s+=0.5*h*ab
        return abs(s)

classificator = BinaryClassificator(porog=160)
classificator.showInputData()

tprate = np.zeros(250)
fprate = np.zeros(250)
precisions = np.zeros(250)
recalls = np.zeros(250)

accuracy_porog = 0
best_porog = 0
best_results_porog = {}

accuracy_probability = 0
best_probability = 0
best_results_probability = {}

for porog in range(0, 250):
    classificatorResult = classificator.porogClassificator(porog)
    tprate[porog] = classificatorResult['tp'] / (classificatorResult['tp'] + classificatorResult['fn'])
    fprate[porog] = classificatorResult['fp'] / (classificatorResult['tn'] + classificatorResult['fp'])
    precisions[porog] = classificatorResult['precision']
    recalls[porog] = classificatorResult['recall']
    if classificatorResult['accuracy']>accuracy_porog:
        accuracy_porog = classificatorResult['accuracy']
        best_porog = porog
        best_results_porog = classificatorResult

classificator.showCurve(fprate,tprate,"ROC Curve")
classificator.showCurve(recalls,precisions,"Positive Rate")
#show porog classificator result
print("=Пороговый классификалор=")
print("Лучший порог:",best_porog,"Accuracy:",accuracy_porog)
print("S под ROC: ",classificator.getSquare(fprate,tprate))
print("S под PR: ",classificator.getSquare(recalls,precisions))
print(best_results_porog)

step=0.01
probabilityLinspace = np.linspace(0,1,1/step)

tprate = np.zeros(len(probabilityLinspace))
fprate = np.zeros(len(probabilityLinspace))
precisions = np.zeros(len(probabilityLinspace))
recalls = np.zeros(len(probabilityLinspace))
for i in range(len(probabilityLinspace)):
    probability = probabilityLinspace[i]

    classificatorResult = classificator.probabilityClassificator(probability=probability)
    tprate[i] = classificatorResult['tp'] / (classificatorResult['tp'] + classificatorResult['fn'])
    fprate[i] = classificatorResult['fp'] / (classificatorResult['tn'] + classificatorResult['fp'])
    precisions[i] = classificatorResult['precision']
    recalls[i] = classificatorResult['recall']

    if classificatorResult['accuracy']>accuracy_probability:
        accuracy_probability = classificatorResult['accuracy']
        best_probability = probability
        best_results_probability = classificatorResult


classificator.showCurve(fprate,tprate,"ROC Curve")
classificator.showCurve(recalls,precisions,"Positive Rate")

print("=Рандомный классификалор=")
print("Лучшая вероятнось:",best_probability,"Accuracy:",accuracy_probability)
print("S под ROC: ",classificator.getSquare(fprate,tprate))
print("S под PR: ",classificator.getSquare(recalls,precisions))
print(best_results_probability)



