"""
In The Name Of God..

@author: Ali Pilehvar Meibody



salam doostan, in porozhe nahayat ta 10 e mehr baraye tahvilash zaman darid

Deadline: 10 e meher

b email e zir ersal farmaeed:
    
    Ai.course22.alipilehvar@gmail.com
    
    
Moafagh Bashid
"""




'''
Ma yek data darim bename boston price

boston yek shahri dar shomale sharghie united state hast
location:https://en.wikipedia.org/wiki/Boston

Dar in shahr omadan hodode 506 ta khone ro yekseri moalefe hash ro
bardashtan neveshtan hamchenin omadan gheymatesho neveshtan.

ma mikhahim ba machine learning azin data yad begirim va azin bebad
har khoone ee dar boston bashe age in chand moalefe ro bede, modelemon
gheymat ro pishbini kone

'''




#baraye load e data kafie benevisid
from sklearn.datasets import load_boston
boston_data=load_boston()



#in data chanta chizi dakhelesh dare
#in ro bznid esme feature ha va vorodi haro mizane k chia hast
boston_data.feature_names


#hala x emon in hast
x=boston_data.data
y=boston_data.target




'''

shoma bayad har 6 ta modele zir ro biad va train bedid

Linear Regression / KNN / Decision Tree / Random Forest / SVR / MLP

hatman gridsearch bezanid ke bebeinid in model ha kodom settingeshon
behtarin javab ro mide

va cross validation kfold 5 bashe , 5 tae bashe va be soorate MAPE gozaresh bedid

bayad MAPE  yek adadi paeen tar az -0.10 bashe va harche paeen tar bashe behtare

dar nahayat yek ax rasm konid ke y= gheymat x=shomare nemone
#bad ye khat k dat ahaye vaghe eie, va bhtrin modeleton, predictionesho rasm konid




'''



'''
Chizi ke baraye man ersal bayad farmaed:
    
1) File e .py e codetoon
2)yek file e excell ke done done model haton ro MAPE ish ro gozaresh dade bashid
    hamrah ba behtarin settingesh ro
    
3)yek file e word az axaee k rasm kardid dar morede moghayese pishbinie behtarin
    modeleton dar barabare dade haye vaghe ei
    
ersal be:
    AI.COURSE22.ALIPILEHVAR@GMAIL.COM
    
DEADLINE: 10 MEHR

Moafagh Bashid.



'''