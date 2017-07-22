import pandas as pd
import numpy as np
import csv
from pandas import Series
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC as Classifier
from sklearn.model_selection import train_test_split
from Preprocessing.create_dataset import get_result
from Preprocessing.createBOW2 import *
import fasttext
from sklearn.metrics import classification_report

class PredictLanguage:

	def __init__(self):
		print "Class created"
		self.data=[]
		self.X_test=pd.DataFrame()
		self.Y_test=pd.DataFrame()
		
		
		self.le = LabelEncoder()
		try:
			self.le.classes_ = np.load('classes.npy')
			
			
		except:
			print "Not found"
	
	
	
	def loadfromCSV(self,filename):
		df = pd.read_csv(filename,index_col=None, header=-1)
		df.dropna(axis=1)
    		self.data.append(df)
		
		
	def storeData(self):
		self.finalData=pd.concat(self.data)
		
		
		
	def preProcess(self):
		var=self.finalData.ix[:,0]
		
		
		
		
		self.finalData.dropna(inplace=True)
		#This adds number of characters per word as a feature
		value=Series([len((str(text).split(' ')))*1.0/len(str(text)) for text in var])
		
		self.finalData=self.finalData.ix[:,1:]
		
		
		#Converts text to number
		self.finalData=self.finalData[self.finalData.columns[:]].apply(self.le.fit_transform)
		
		np.save('classes.npy', self.le.classes_)
		
		self.finalData.ix[:,4]=value
		
		
		
		
		
	def Train(self):
		X=self.finalData.ix[:,(1,2,4)]
		
		Y=self.finalData.ix[:,3]
		
		
		
		self.clf = Classifier()
		self.clf = self.clf.fit(X,Y)
		
		
		
		
	
	def Test(self,fileName,outFile):
			
		model = fasttext.supervised('Preprocessing/dataFile2.txt','mymodel')
		TestData=pd.DataFrame()
		csvfile=open(fileName,'rb')
		spamreader = csv.reader(csvfile, delimiter=',')
		c=0
		for row in spamreader:
			print c
			try:
				lenparam=len((str(row[0]).split(' ')))*1.0/len(str(row[0]))
			except:
				continue
			first=str(start(row[0])).strip()
			second=str(get_result(row[0],model)).strip()	
			output=row[1].strip()
			if c==1:
				print first,second,output
			TestData=TestData.append(pd.DataFrame([[first,second,lenparam,output]]),ignore_index=True)
			if c > 500:
				break
			c=c+1	
				
		
		
		tempDF=TestData.ix[:,(0,1,3)]
		
		
		
		#Uncomment this
		tempDF=tempDF[tempDF.columns[:]].apply(self.le.transform)
		
		
		TestData.ix[:,(0,1,3)]=tempDF
		
		print self.le.classes_
		
		TestData.to_csv(outFile,ignore_index=True)
		self.X_test=self.X_test.append(TestData.ix[:,(0,1,2)],ignore_index=True)
		self.Y_test=self.Y_test.append((TestData.ix[:,3]),ignore_index=True)
		
		
		
			
	def ShowAccuracy(self):
		filenames=['Test/testData.csv','Test/testData2.csv','Test/testData3.csv']
		all_test_data=pd.DataFrame()
		for each_file in filenames:
			df = pd.read_csv(each_file,index_col=None, header=-1)
			df.dropna(axis=1)
			
	    		all_test_data=all_test_data.append(df[1:],ignore_index=True)
	    		
	    		
	    	X_test=all_test_data.ix[:,(1,2,3)]
	    	Y_test=all_test_data.ix[:,4]	
	    	print "(Precision,Recall and FScore)"
	    	
	    	print(classification_report(Y_test,self.clf.predict(X_test), target_names=self.le.classes_)) 
	    	#print X_test.tail()
	    	#print self.clf.predict(X_test.tail())
	    	print self.clf.score(X_test,Y_test)
	
	
	def ActuallyTest(self,testString):
		model = fasttext.supervised('Preprocessing/dataFile2.txt','mymodel')
		lenparam=len((str(testString).split(' ')))*1.0/len(str(testString))
		first=str(start(testString)).strip()
		second=str(get_result(testString,model)).strip()	
		
		TestData=pd.DataFrame([[first,second,lenparam]])
		tempDF=TestData.ix[:,(0,1)]
		tempDF=pd.DataFrame(self.le.transform(tempDF.values.ravel()).reshape(tempDF.shape), columns =tempDF.columns)
		TestData.ix[:,(0,1)]=tempDF
		
		return self.clf.predict(TestData)
			
				 
a=PredictLanguage()	

a.loadfromCSV('finaltrain/out_train.csv')
a.loadfromCSV('finaltrain/out_train_en.csv')
a.loadfromCSV('finaltrain/out_train_np2.csv')
a.storeData()
a.preProcess()
a.Train() 
#a.Test('Test/nepali_sent2.txt','Test/testData3.csv')
#a.Test('Test/eng_sents.txt','Test/testData2.csv') 
#a.Test('Test/output.txt','Test/testData.csv')
a.ShowAccuracy()
test=''	
while (test!="quit"):
	test=raw_input("Enter text to classify: ")
	test=test.replace(',','')
	print a.le.inverse_transform(a.ActuallyTest(test))
