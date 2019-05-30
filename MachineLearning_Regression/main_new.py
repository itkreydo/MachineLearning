import numpy as np
import matplotlib.pyplot as plt

class Regression:
    def __init__(self,N=1000,M_MAX=100):
        self.N=N
        self.M=M_MAX
        self.x=np.linspace(0,1,N)
        self.z = 20 * np.sin(2 * np.pi * 3 * self.x) + 100 * np.exp(self.x)
        e=10*np.random.randn(N)
        self.t=self.z+e
        self.stylePlotData="yo"
        self.stylePlotTrend ="-r"
        self.stylePlotTrend2 = "-b"

    def showInputData(self):
        self.showGraphic(self.x, self.z,onlyData=True, title="Входные данные x,z,t")

    def validationSet(self,showPlotWithM=[1,8,100]):
        middleError = np.zeros(self.M+1)
        mx = np.arange(0,self.M+1)
        for m in range(0,self.M+1):
            self.getDesignMatrix(m)
            y = self.getValidationData(m)
            for i in range(self.N):
                middleError[m]+=0.5*(self.t[i]-y[i])**2
            if m in showPlotWithM:
                self.showGraphic(self.x, y, title="График, при M = "+str(m))
        self.showGraphic(mx, middleError, showData=False, title="График ошибки")

    def getValidationData(self,M):
        return self.getWparams()@self.designMatrix.T

    def getDesignMatrix(self,M):
        matrixPlanF = np.zeros((self.N,M+1),dtype=np.float)
        matrixPlanF[:,0]=1
        if M!=0:
            for i in range(1,M+1):
                matrixPlanF[:, i] = self.x.T**i
        self.designMatrix=matrixPlanF
        return matrixPlanF

    def getWparams(self):
        return np.linalg.inv(self.designMatrix.T@self.designMatrix)@self.designMatrix.T@self.t

    def showGraphic(self,x,y,onlyData=False,showData=True,title="График"):
        plt.figure()
        if showData:
            plt.plot(self.x, self.t, self.stylePlotData,markersize=2)
            plt.plot(self.x, self.z, self.stylePlotTrend)
        if not onlyData:
            plt.plot(x, y, self.stylePlotTrend2)
        plt.title(title)
        plt.show()


linReg = Regression(N=1000,M_MAX=1000)
linReg.showInputData()
linReg.validationSet(showPlotWithM=[1,8,100])


