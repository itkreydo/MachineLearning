import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import load_digits

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.split_ind = None
        self.split_tao = None
        self.h_value = None
        self.prob_vec = None

class Decision_tree:
    def __init__(self, dept_max, n_min, h_min):
        self.R = 100
        self.root = Node()
        self.dept_max = dept_max
        self.n_min = n_min
        self.h_min = h_min

    def fit(self,x,t):
        start_depth=0
        self.build( x, t, start_depth, self.root)
    def predict(self,x):
        res =[]
        for i in range(len(x)):
            y_i = self.predict_one(x[i],self.root)
            res.append(y_i)
        return np.array(res)
    def isNone(self,a):
        if (a is None):
            return True
        return False
    def predict_one(self, x, nodeCurrent):
        if (not self.isNone(nodeCurrent.prob_vec)):
            return nodeCurrent.prob_vec
        if (x[nodeCurrent.split_ind]>nodeCurrent.split_tao):
            return self.predict_one(x, nodeCurrent.right)
        else:
            return self.predict_one(x, nodeCurrent.left)


    def P(self,t):
        p = sum(t[:])/len(t)
        return p
    def H(self,t):
        h = 0
        #print("sum()=", (sum(t[:])/len(t)), "x in log =",sum(t[:])/len(t),"log = ", np.log(sum(t[:])/len(t)))
        #print("sum()=",(sum(t[:])/len(t))*np.log(sum(t[:])/len(t)))

        for i in range(len(t[0])):
            h_tmp = (np.sum(t[:,i])/len(t))*np.log(np.sum(t[:,i])/len(t))
            if (not np.isnan(h_tmp)):
                h-= h_tmp
        return h
    def IG(self,t_i,t_j_left,t_j_right):
        try:
            #print("H(ti)=",self.H(t_i),"len del = ",len(t_j_left)/len(t_i),"Hleft =",self.H(t_j_left),"Hright =",self.H(t_j_right))
            ig = self.H(t_i) - (len(t_j_left)/len(t_i))*self.H(t_j_left) - (len(t_j_right)/len(t_i))*self.H(t_j_right)
        except:
            ig=0
        return ig
    def split_arr(self,x,t,ind,tao):
        x_left = []
        x_right = []
        t_right = []
        t_left = []
        for i in range(len(x)):
            if x[i][ind] > tao:
                t_right.append(t[i])
                x_right.append(x[i])
            else:
                t_left.append(t[i])
                x_left.append(x[i])

        return {"x_left":x_left,"x_right":x_right,"t_left":np.array(t_left),"t_right":np.array(t_right)}
        pass
    def build(self, x, t, depth, nodeCurrent):

        if (depth == self.dept_max or len(x) < self.n_min or self.H(t) < self.h_min):
            #print(depth,"len(x)=", len(x) , "n_min=",self.n_min, "H(t) =",self.H(t) ,"h_min=", self.h_min)
            nodeCurrent.prob_vec = self.P(t)

            return
        IG_prev=0
        optimal_ind=0
        optimal_tao=0
        for i in range(self.R):
            ind = np.random.randint(64)
            tao = np.random.randint(16)
            splited_arr = self.split_arr(x,t,ind,tao)#left/right
            IG_cur = self.IG(t, splited_arr["t_left"],splited_arr["t_right"])

            if (IG_cur>IG_prev):
                IG_prev = IG_cur
                optimal_ind = ind
                optimal_tao = tao
        nodeCurrent.split_ind = optimal_ind
        nodeCurrent.split_tao = optimal_tao

        splited_arr = self.split_arr(x,t,optimal_ind, optimal_tao)
        if len(splited_arr["x_left"])>0:
            node_left = Node()
            nodeCurrent.left = node_left
            self.build(splited_arr["x_left"],splited_arr["t_left"],depth+1,node_left)
        if len(splited_arr["x_right"]) > 0:
            node_right = Node()
            nodeCurrent.right = node_right
            self.build(splited_arr["x_right"],splited_arr["t_right"],depth+1,node_right)







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
        self.trainData = self.trainData
        self.testData = self.testData
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







class RandomForest:
    def __init__(self,num_trees=100):
        self.num_trees = num_trees
        self.trees = []
    def genForestFit(self,x,t,depth_max,n_min,h_min):
        for i in range(self.num_trees):
            tree = Decision_tree(depth_max,n_min,h_min)
            #tree.fit(x,t)
            self.trees.append(tree)
            self.trees[i].fit(x,t)
    def ForestPredict(self,x,t):
        predict_massive = []
        for i in range(self.num_trees):
            predict_massive.append(self.trees[i].predict(x))

        predict_massive = np.array(predict_massive)
        result = []

        for i in range(len(x)):
            result.append(sum(predict_massive[:,i])/self.num_trees)

        self.showResultMetrix(result,t)
        return result
    def getAccuracy(self,t_predicted,t_right):
        TP = 0
        N = len(t_predicted)
        for i in range(len(t_predicted)):
            if (np.argmax(t_predicted[i]) == np.argmax(t_right[i])):
                TP += 1
        return TP/N
    def getConfusionMatrix(self,t_predicted,t_right):
        confusionMatrix = np.zeros((10, 10))
        for i in range(len(t_predicted)):
            confusionMatrix[np.argmax(t_predicted[i])][ np.argmax(t_right[i])] += 1
        return confusionMatrix
    def showResultMetrix(self,t_predicted,t_right):
        print("Accuracy:", self.getAccuracy(t_predicted,t_right))
        print("ConfusionM:", self.getConfusionMatrix(t_predicted,t_right))



dg = DigitsClassificator()

randomForest = RandomForest(num_trees=10)
print("==trainSet==")
randomForest.genForestFit(dg.trainData,dg.trainTarget,7,1,0.2)
randomForest.ForestPredict(dg.trainData,dg.trainTarget)
print("==testSet==")
predictions= randomForest.ForestPredict(dg.testData,dg.testTarget)







