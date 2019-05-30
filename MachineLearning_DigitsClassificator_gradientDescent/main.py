import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import load_digits

class DigitsClassificator:
    def __init__ (self,train_set=0.8):
        self.digits = load_digits()
        self.N = len(self.digits.data)
        #shuffle data
        ind = np.arange(self.N)
        np.random.shuffle(ind)
        indTrain = ind[:int(self.N*train_set)]
        indTest = ind[int(self.N*train_set):]
        self.testData = np.array(self.digits.data[indTest],dtype=np.float64)
        self.testTarget = self.oneHotEncoding(self.digits.target[indTest])
        self.testImage = self.digits.images[indTest]
        self.trainData = np.array(self.digits.data[indTrain],dtype=np.float64)
        self.trainTarget = self.oneHotEncoding(self.digits.target[indTrain])
        self.trainImage = self.digits.images[indTrain]
        #standartizize
        #print(np.max(sk.preprocessing.scale(self.trainData)))
        self.trainData = self.standartisation(self.trainData)
        self.testData = self.standartisation(self.testData)
        #end shuffled data

    def oneHotEncoding(self,targetArray):
        newTarget = np.zeros([len(targetArray),10])
        for i in range(len(targetArray)):
            newTarget[i][targetArray[i]]=1
        return newTarget

    def standartisation(self,data):

        U = np.zeros(64)#len(self.trainData[0])
        sigma = np.zeros(64)
        for i in range(64):
            U[i] = np.sum((data[:,i]/len(data)))
        for i in range(64):
            sigma[i] = np.sqrt(np.sum((data[:,i]-U[i])**2)/len(data))
        for i in range(len(data)):
            for j in range(64):
                data[i][j] = 0 if (sigma[j]==0) else (data[i][j] - U[j])/sigma[j]

        return data

    def init_Wb(self,type="Normal",M=0,sigma=1,eps = 0.005,k=10,d=64):
        #np.finfo(np.float32).eps**2
        if (type=="Normal"):
            W = np.random.normal(M, sigma, [k,d])
        if (type=="Uniform"):
            W = np.random.uniform(-eps, eps, [k,d])
        if (type=="Xavior"):
            W = np.random.uniform(-eps, eps, [k,d])
        if (type=="He"):
            W = np.random.uniform(-eps, eps, [k,d])
        return W

    def softmax(self,y):
        e_x = np.exp(y - np.max(y))
        return e_x / e_x.sum()
    def softmax_all(self,y_vector):
        for i in range(len(y_vector)):
            y_vector[i] = self.softmax(y_vector[i])
        return y_vector
    def getError(self,t,y):
        Error=0
        for i in range(len(y)):
            for k in range(len(y[0])):
                Error -= t[i,k]*np.log(y[i,k])

        return Error

    def getAccuracy(self,t,y):
        max_e = y.argmax(axis=1)
        acc=0
        for i in range (len(max_e)):
            acc+= 1 if (t[i,max_e[i]]==1) else 0

        # acc = np.sum(t[:, max_e]) / len(max_e)
        return acc/len(max_e)

    def trainSetLearning(self,stop_iter=False,iter=1000):
        self.W = self.init_Wb()
        self.b = self.init_Wb(d=1)
        self.W_best = self.W
        self.b_best = self.b
        U = np.ones((len(self.trainData),1))
        stop_status=True

        lyambda = 0.001
        self.lyambda = lyambda
        gamma = 0.05
        eps = 0.05
        i=0
        E_array = []
        E_accuracy = []
        E_test = []
        A_test = []
        AccTestMax =0
        while (stop_status):
            i+=1
            y_train = self.softmax_all((self.W@self.trainData.T + self.b).T)
            Ew_gradient = (y_train-self.trainTarget).T@(lyambda*self.trainData)
            Eb_gradient = (y_train - self.trainTarget).T @ U
            W_new = self.W - gamma*Ew_gradient
            W_b = self.b - gamma*Eb_gradient
            E = self.getError(self.trainTarget,y_train)
            Acc = self.getAccuracy(self.trainTarget,y_train)
            ##############################
            ##test data (ERROR ACCURACY)##
            testSetData = self.testSet(self.W,self.b)
            E_test.append(testSetData["Error"])
            A_test.append(testSetData["Accuracy"])

            if (A_test[i-1]>AccTestMax):
                self.W_best=self.W
                self.b_best=self.b
                AccTestMax=A_test[i-1]#current accurecy test

            #############################
            if (np.sum(np.abs(W_new-self.W))<eps and np.sum(np.abs(W_b-self.b))<eps and not stop_iter):
                stop_status=False
            if ( i == 1000 and stop_iter):
                stop_status=False


            self.W = W_new
            self.b = W_b

            E_array.append(E)
            E_accuracy.append(Acc)
            if (i%15==0):
                print("i:",i,"\nE_train =",E,"Acc_train =",Acc, "\nE_test =",testSetData["Error"],"Acc_tset =",testSetData["Accuracy"])
        print("count of iter:",i)

        self.showGraphic(np.arange(len(E_array)),E_array,title="Ошибка Train")
        self.showGraphic(np.arange(len(E_accuracy)), E_accuracy, title="Accuracy Train")
        self.showGraphic(np.arange(len(E_test)), E_test, title="Ошибка Test")
        self.showGraphic(np.arange(len(A_test)), A_test, title="Accuracy Test")
        print("AccTestMax",AccTestMax)

    def testSet(self,w_best,b_best,print_result=False,show_image=False):
        y_test = self.softmax_all((w_best @ self.testData.T + b_best).T)
        E = self.getError(self.testTarget, y_test)
        Acc = self.getAccuracy(self.testTarget,y_test)
        if (show_image):
            self.getBadAndBestImages(self.testTarget,y_test)

        if (print_result):
            print("E test:",E)
            print("Acc test:", Acc)
        return {"Error":E,"Accuracy":Acc}
            # Eb_gradient = (y - self.trainTarget).T @
    def showGraphic(self, x, y, onlyData=False, showData=True, title="График"):
        plt.figure()
        plt.plot(x, y, "-r")
        plt.title(title)
        plt.show()
    def getBadAndBestImages(self,t,y,n = 3):
        best=[]
        bad=[]
        max_e = y.argmax(axis=1)
        acc=0
        for i in range (len(max_e)):
            if (t[i,max_e[i]]==1):
                best.append([y[i][max_e[i]],i])
            else:
                bad.append([y[i][max_e[i]],i])
        best = np.array(best)
        bad = np.array(bad)
        best = (best[best[:, 0].argsort()])[::-1]
        bad = (bad[bad[:, 0].argsort()])[::-1]
        for i in range(n):
            plt.figure()
            # print(np.where(self.testTarget[int(best[i,1])]==1))
            plt.title("Лучшие картинки #"+str(i+1))
            plt.imshow(self.testImage[int(best[i,1])])
            plt.show()
        for i in range(n):
            plt.figure()
            # print(np.where(self.testTarget[int(best[i, 1])] == 1))
            plt.title("Худшие картинки #"+str(i+1))
            plt.imshow(self.testImage[int(bad[i,1])])
            plt.show()
        print(best,bad)



dg = DigitsClassificator()
dg.trainSetLearning(stop_iter=False)#При stop_iter = False Спуск прекращается по условию ||Wk+1-Wk||<eps, если True, то по количеству итераций, доп агрумент iter=1000
print("==testSet==")
dg.testSet(dg.W_best,dg.b_best,print_result=True,show_image=True)
