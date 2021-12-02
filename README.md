# MedicalImageTamperDetection
Team members: **Zack Muraca, Gary Kim, Thongsavik Sirivong, Nhat Tran**

## Purpose

Today, a CT (Computed Tomography) scanner is an essential tool in health-care systems. CT scans are used to diagnose certain diseases (e.g., heart disease, trauma, infectious diseases, cancer, etc.) In 2020, health-care systems were attacked brutally leading to treatment interruptions and medical record breaches. Concerning the influential aspects of medical data, we will have to focus on inproving the security of health-care systems. In this research, we use a deep learning machine learning algorithm to detect tampered medical images.

## Task Description

* [x]  get data and transform into matrix
* [x]  Run different algs (regression, ensemble, decision tree, etc.)
* [x]  Run neural networks
* [x]  Record results
* [x]  Related work
* [x]  Tidy up final report

## Results

Our initial model with a regular deep learning architecturehad  an  accuracy  of   30%  on  the  test  data.  Since  it  was  onlytrained  on10%  of  the  full  dataset  and  the  test  set  camefrom  a  different  distribution  than  the  training  set,  this  resultwas expect. Even though random guessing has an accuracy of50%,  the  model  was  fit  to  the  training  data  and  was  unableto  generalize  at  all  to  the  test  set.  Our  3D  CNN  model  wastrained on the full data for 200 epochs. This model was tunedfor accuracy on the validation set, which comes from the samedistribution as the test set. It had an accuracy of   50% on thetraining set,   75% on the validation set, and 70% on the testset.The evaluations done in the open trials of’CT-GAN: Mali-cious tampering of 3D medical imagery using deep learning’involved 4 detectors (radiologists of 2, 5, 7 years of experienceand  AI)  locating  areas  of  true/false  cancers  and  true/falsebenign areas. The average false positive (detected fake cancer)rate  was  99.2%  and  the  average  false  negative  (detected  nocancer given real cancer was removed) 95.8%.This  task  is  much  more  complex  than  the  binary  classifi-cation task of our research. However, since we have no otheralternatives  for  an  optimal  classifier  proxy,  we  will  compareour results to this accordingly.In  comparison,  our  70%  accuracy  for  our  first  3D  CNNmodel  looks  promising.  It  is  possible  that  further  tuning  andrefinement  of  the  same  model  can  make  improvements  onthis number. However, we believe taking an entirely differentapproach would be better. Our model is tasked with classifyingtampered images given that the image has no cancers and hasmany cancers. The image looks completely different in benignscans  versus  cancerous  scans.  Our  model  may  be  havingtrouble in detecting tampering because areas of tampering maylook  significantly  different  in  healthy  and  unhealthy  scans.This  was  our  observation  after  working  with  the  data,  butan in-depth texture analysis of the images would most likelyprovide key insights.Separating the full tamper detection into three tasks makesmore  sense  intuitively  and  should  be  the  focus  of  futureresearch.  The  first  task  is  cancer  detection  followed  by  thesecond  task  of  tamper  detection,  therefore  the  first  model  isfor detecting malignant or benign cancer. One tamper detectionmodel would learn to detect tampering given cancers. Lastly,another  model  would  learn  to  detect  tampering  given  nocancers.

## Conclusion

Some takeaways from this research were the level of com-mitment required to effectively undertake such a task and pro-duce results, initial experience required for breaking into thisarea of machine learning, and knowledge of popular machinelearning  tools  such  as  Keras  and  Tensorflow.  Additionally,we  gained  familiarity  with  the  GCP  environment  where  inwhich we utilized an Nvidia Tesla K80 GPU to speed up theprocess. Each of these aspects are invaluable for contributingto research in the medical field.As  mentioned  before,  we  will  continue  our  research  byinvestigating  the  three-task  solution  proposed.  It  is  uncertainat  this  moment  whether  it  will  pay  off,  but  it  appears  to  bethe next best objective for us moving forward.

## GitHub repository

[GitHub repository](https://github.com/viksirivong/MedicalImageTamperDetection)



