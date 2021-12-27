import bayes,sys,time,bayesbest

if __name__ == '__main__':

	# print(len(sys.argv))
	
	if len(sys.argv) == 4:
		startTime=time.time()
		if "bayes" in sys.argv:
			
			b=bayes.Bayes_Classifier(sys.argv[2]+"/")
			print("Time to train: ",time.time()-startTime)
			testTime=time.time()
			b.testBayes(sys.argv[3]+"/")
			print("Time to classify: ",time.time()-testTime)
		elif "bayesbest" in sys.argv:
			
			b=bayesbest.Bayes_Classifier(sys.argv[2]+"/")
			print("Time to train: ",time.time()-startTime)
			testTime=time.time()
			b.testBayes(sys.argv[3]+"/")
			print("Time to classify: ",time.time()-testTime)
		else:
			print("Please enter the name of the classifier you wish to run as part of cmd argument.[bayes or bayesbest]")

	else:
		print("Incorrect arguments for evaluate.py.Please run: sh run.sh [Classifier name] [Train directory] [Test directory]")
		print("If pickled files exist, classification happens directly. To retrain, delete files created.")
\

    