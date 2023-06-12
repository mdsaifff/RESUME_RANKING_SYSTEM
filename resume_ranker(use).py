# -*- coding: utf-8 -*-

!pip install docx2txt

!pip install textract

import tensorflow as tf
import tensorflow_hub as hub 
import numpy as np
import docx2txt
import glob
import re

module_url= "https://tfhub.dev/google/universal-sentence-encoder/4"
model= hub.load(module_url)
print("Module %s loaded" %module_url)

job_post_path='/content/drive/MyDrive/Resume_parser/Job_post/Machine_learning_JD.docx'
job_post= docx2txt.process(job_post_path)

job_post

all_files=glob.glob('/content/drive/MyDrive/Resume_parser/Resume_for_test/*', recursive= True)

def cosine(u,v):
  return np.dot(u,v)/ (np.linalg.norm(u) * np.linalg.norm(v))

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  
    resumeText = re.sub('RT|cc', ' ', resumeText)  
    resumeText = re.sub('#\S+', '', resumeText) 
    resumeText = re.sub('@\S+', '  ', resumeText) 
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  
    return resumeText

score_list=[]
file_name=[]
for single_file in all_files:
  my_text = docx2txt.process(single_file)
  #my_text= cleanResume(my_text)
  sentences_list=[my_text]
  #print(sentences_list)
  #sentence_embeddings = model(sentences_list)
  jd_vec = model([job_post])[0]
  for sent in sentences_list:
      
      similarity = cosine(jd_vec, model([sent])[0])
      score_list.append(similarity)
      file_name.append(single_file)

      #print("File = ", single_file, " \tsimilarity = ",similarity)

mapped = zip(file_name, score_list)
#print (mapped)
mapped = list(mapped)

result = sorted(mapped, key = lambda x: x[1],reverse=True)
#print (result)

for i in result:
  print(i)





