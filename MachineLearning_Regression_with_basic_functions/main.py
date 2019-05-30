import numpy as np
import matplotlib.pyplot as plt

class Regression:
    def __init__(self,N=1000):
        self.N=N
        self.x=np.linspace(0,1,N)
        self.z = 20 * np.sin(2 * np.pi * 3 * self.x) + 100 * np.exp(self.x)
        e=10*np.random.randn(N)
        self.t=self.z+e
        #divide generated data
        self.divideTrainValidTestData(train_percent=0.8, valid_percent=0.1)#test_percent=1 - train_percent - valid_percent
        #plot settings
        self.stylePlotData="yo"
        self.stylePlotTrend ="-r"
        self.stylePlotTrend2 = "-b"

    def divideTrainValidTestData(self,train_percent=0.8,valid_percent=0.1,test_percent=0.1):
        indexes = np.arange(self.N)
        indexes_per = np.random.permutation(indexes)
        ind_train_percent = np.int32(train_percent*len(indexes_per))#80%
        ind_valid_percent = np.int32((train_percent+valid_percent) * len(indexes_per))#90%
        ind_train = indexes_per[:ind_train_percent]
        ind_valid = indexes_per[ind_train_percent:ind_valid_percent]
        ind_test = sorted(indexes_per[ind_valid_percent:])
        self.train_x = self.x[ind_train]
        self.train_t = self.t[ind_train]
        self.valid_x = self.x[ind_valid]
        self.valid_t = self.t[ind_valid]
        self.test_x = self.x[ind_test]
        self.test_t = self.t[ind_test]
    def initValidationData(self):
        self.lambda_vals = [0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
        self.funcs_list = [
                      lambda x:1,
                      lambda x:x,
                      lambda x:x**2,
                      lambda x:x**3,
                      lambda x:x**4,
                      lambda x:x**5,
                      lambda x: x ** 6,
                       lambda x: x ** 7,
                       lambda x: x ** 8,
                       lambda x: x ** 9,
                       lambda x: x ** 10,
                       lambda x:np.exp(x),
                       lambda x:np.sin(x),
                       lambda x: np.sin(2*x),
                       lambda x:np.cos(x),
                       lambda x: (np.cos(x)*np.sin(x))**2,
                       lambda x:np.tan(x),
                       lambda x:np.sqrt(x)]
        self.E_min = 10**10
        self.funcs_list_text =['1','x','x^2','x^3','x^4','x^5','x^6','x^7','x^8','x^9','x^10','exp(x)','sin(x)','sin(2x)','cos(x)','(cos(x)*sin(x))^2','tan(x)','sqrt(x)']
        pass
    def validationSet(self,showPlotWithM=[1,8,100],iters_num=1000):
        self.initValidationData()#input_data
        for i in range(iters_num):
            phiCur = self.get_funcs(self.funcs_list)
            lyambdaCur = self.get_lyambda(self.lambda_vals)
            FTrain = self.getDesignMatrix(phiCur,self.train_x)#design matrix
            try:
                wCur = self.getWparams(FTrain, self.train_t, lyambdaCur)
            except:
                #print("determinant=0 continue")
                continue
            FValid = self.getDesignMatrix(phiCur,self.valid_x)
            E_cur = 0.5*np.sum((self.valid_t - FValid@wCur)**2)
            if E_cur < self.E_min:
                self.phiBest=phiCur
                self.lyambdaBest=lyambdaCur
                self.E_min=E_cur
                self.wBest = wCur
        print("Phi_best: ", self.getTextFuncList(self.phiBest))
        print("Lyambda best:", self.lyambdaBest)
    def get_funcs(self,funcs_list):
        func_num = np.random.randint(1,len(funcs_list)+1)
        return np.random.choice(funcs_list,func_num,replace=False)
    def getTextFuncList(self,func_list):
        func_text=[]
        for func in func_list:
            func_text.append(self.funcs_list_text[self.funcs_list.index(func)])
        return func_text
    def get_lyambda(self,lyambda_list):
        return np.random.choice(lyambda_list)
    def getValidationData(self,w,designMatrix):
        return w@designMatrix.T
    def getDesignMatrix(self,phi_cur,x):
        matrixPlanF = np.zeros((len(x),len(phi_cur)+1))
        matrixPlanF[:,0]=1
        for i in range(1,len(phi_cur)+1):
            matrixPlanF[:, i] = phi_cur[i-1](x)
        self.designMatrix=matrixPlanF
        return matrixPlanF
    def getWparams(self,designMatrix,t,lyambda):
        I = np.eye(len(designMatrix.T))
        return np.linalg.inv(designMatrix.T@designMatrix + lyambda*I)@designMatrix.T@t
    def testSet(self):
        print("===Start test set===")
        F_test = self.getDesignMatrix(self.phiBest, self.test_x)
        y = self.getValidationData(self.wBest, F_test)
        E_test = 0.5 * np.sum((self.test_t - y) ** 2)
        print("E_test: ",E_test)
        self.showGraphic(self.test_x, y, title="График")
    def showGraphic(self,x,y,onlyData=False,showData=True,title="График"):
        plt.figure()
        if showData:
            plt.plot(self.train_x, self.train_t, self.stylePlotData, markersize=2)
            plt.plot(self.valid_x, self.valid_t, self.stylePlotData, markersize=2)
            plt.plot(self.test_x, self.test_t, "go",markersize=2)
            plt.plot(self.x, self.z, self.stylePlotTrend)
        if not onlyData:
            plt.plot(x, y, self.stylePlotTrend2)
        plt.title(title)
        plt.show()


linReg = Regression(N=1000)
linReg.validationSet(iters_num=10000)
linReg.testSet()


