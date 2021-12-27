import math, os, pickle, re,time,sys



Ratings={0:"Negative",1:"Negative",2:"Negative",3:"Positive",4:"Positive",5:"Positive"}
class Bayes_Classifier:

    def __init__(self, trainDirectory = "movies_reviews/"):
        '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
        cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
        the system will proceed through training.  After running this method, the classifier 
        is ready to classify input text.'''
        self.trainDirectory=trainDirectory
        IfileList=[]
        countPositive=0
        countNegative=0
        for fileObj in os.walk(self.trainDirectory):
            IfileList=fileObj[2]
        
        for fileName in IfileList:
            # print(fileName)
            match = re.search(r"(\d)",fileName)
            rating=(int)(match.group())
            if rating in (4,5):
                countPositive+=1
            else:
                countNegative+=1
        
        # countNegative*3
        # print("//",countPositive,countNegative)
        self.priorPositive=countPositive/(countPositive+countNegative)
        self.priorNegative=countNegative/(countPositive+countNegative)


        if os.path.exists("positive_reviews") and os.path.exists("negative_reviews") :
            self.dictPositive=self.load("positive_reviews")
            self.dictNegative=self.load("negative_reviews")
          
        else:
            self.dictPositive={}
            self.dictNegative={}
            self.save(self.dictPositive,"positive_reviews")
            self.save(self.dictNegative,"negative_reviews")
     
        if not len(self.dictPositive) and not len(self.dictNegative):
            self.train()
        
        

    def train(self):   
        '''Trains the Naive Bayes Sentiment Classifier.'''
        # print(self.dictPositive,self.dictNegative)
        IfileList=[]
        
        for fileObj in os.walk(self.trainDirectory):
            IfileList=fileObj[2]
       
        for fileName in IfileList:
         
            match = re.search(r"(\d)",fileName)
            rating=(int)(match.group())
            sText=self.loadFile(self.trainDirectory + fileName)
           
            tText=self.tokenize(sText)


            # Store Word Count for Unigrams
            for word in tText:
                word=word.lower()
                if word in (".","!","?",","):
                    continue
                if rating in (4,5):
                    if word in self.dictPositive.keys():
                        self.dictPositive[word]+=1
                    else:
                        self.dictPositive[word]=1
                else:
                    if word in self.dictNegative.keys():
                        self.dictNegative[word]+=1
                    else:
                        self.dictNegative[word]=1
  
        featureFrequencyPositive=0
        featureFrequencyNegative=0
   
        # Store Probability values and pickle
        for i in self.dictPositive.values():
            featureFrequencyPositive+=i
        for i in self.dictNegative.values():
            featureFrequencyNegative+=i

        #Add One Smoothening
        for word in self.dictPositive.keys():
            if word in self.dictNegative.keys():
                self.dictPositive[word]=((self.dictPositive[word]/featureFrequencyPositive))
            else:
                self.dictPositive[word]=(((self.dictPositive[word]+1)/(featureFrequencyPositive+1)))
                self.dictNegative[word]=((1/(featureFrequencyNegative+1)))
                featureFrequencyPositive+=1
                featureFrequencyNegative+=1
        for word in self.dictNegative.keys():
            if word in self.dictPositive.keys():
                self.dictNegative[word]=((self.dictNegative[word]/featureFrequencyNegative))
            else:
                self.dictNegative[word]=(((self.dictNegative[word]+1)/(featureFrequencyNegative+1)))
                self.dictPositive[word]=((1/(featureFrequencyPositive+1)))
                featureFrequencyPositive+=1
                featureFrequencyNegative+=1
                    

        #pickle
        self.save(self.dictPositive,"positive_reviews")
        self.save(self.dictNegative,"negative_reviews")

            

            #  print(rating)


        
    def classify(self, sText):
        '''Given a target string sText, this function returns the most likely document
        class to which the target string belongs. This function should return one of three
        strings: "positive", "negative" or "neutral".
        '''
       
        condProbabilityPositive=math.log(self.priorPositive)
        condProbabilityNegative=math.log(self.priorNegative)
       
        tText=self.tokenize(sText)
        # print(tText)
        for word in tText:
            # applying add one Laplace's Smoothening
            if word in self.dictPositive.keys():
                condProbabilityPositive+=math.log(self.dictPositive[word])
            if word in self.dictNegative.keys():
                condProbabilityNegative+=math.log(self.dictNegative[word])       
                    
           
          
        probabilityPositive= condProbabilityPositive
        probabilityNegative= condProbabilityNegative
     

        if probabilityPositive>probabilityNegative:
            return "Positive"
        else:
            return "Negative"
    
            



    def loadFile(self, sFilename):
        '''Given a file name, return the contents of the file as a string.'''

        f = open(sFilename, "r")
        sTxt = f.read()
        f.close()
        return sTxt
    
    def save(self, dObj, sFilename):
        '''Given an object and a file name, write the object to the file using pickle.'''

        f = open(sFilename, "wb+")
        p = pickle.Pickler(f)
        p.dump(dObj)
        f.close()
    
    def load(self, sFilename):
        '''Given a file name, load and return the object stored in the file.'''

        f = open(sFilename, "rb")
        u = pickle.Unpickler(f)
        dObj = u.load()
        f.close()
        return dObj

    def tokenize(self, sText): 
        '''Given a string of text sText, returns a list of the individual tokens that 
        occur in that string (in order).'''

        lTokens = []
        sToken = ""
        for c in sText:
            if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
                sToken += c
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip()))
                
        if sToken != "":
            lTokens.append(sToken)

        return lTokens

    def checkTokenizer(self,fileName):
        sText=self.loadFile(fileName)
        print(sText)
        #   for word in text.split():
        print(self.tokenize(sText))
            
            
        return
    
    #function to train and test directories
    def testBayes(self,folderName):

    

        for fileObj in os.walk(folderName):
            IfileList=fileObj[2]
        trueCount=0
        falseCount=0
        truePositive=0
        trueNegative=0
        falseNegative=0
        falsePositive=0
        for fileName in IfileList:
            # print(fileName)
            match = re.search(r"(\d)",fileName)
            rating=(int)(match.group())
            # print(rating,fileName)
            rating=Ratings[rating]
            sText=self.loadFile(folderName + fileName)
            # print(sText)
            review=self.classify(sText)

            if rating==review:
                trueCount+=1
                if rating=="Positive":
                    truePositive+=1
                elif rating=="Negative":
                    trueNegative+=1
                
            else:
                if rating=="Positive":
                    falseNegative+=1
                elif rating=="Negative":
                    falsePositive+=1
              
                falseCount+=1
        
        print("True Positives: ",truePositive)
        print("True Negatives: ",trueNegative)
        print("Recall: ",truePositive*100/(truePositive+falseNegative))
        print("Precision: ",truePositive*100/(truePositive+falsePositive))
        print("Accuracy: ",(trueCount/(trueCount+falseCount))*100)
        return
       

# b=Bayes_Classifier("Train/")
# print(b.classify("Worst movie.Overrated"))
# print(b.classify("I love my AI class")) 
# print(b.classify("The potrayal of the character by the actors was out of this world")) 
# print(b.classify("not good at all")) 
# print(b.classify("the acting was not above the norms. "))
# print(b.classify("The performance by Dicaprio really made me hate his character in the movie. Slavery,Racism and its consequences potrayed"))  

      
