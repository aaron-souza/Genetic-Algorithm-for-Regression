import pandas as pa
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as py

#Import csv file
data = pa.read_csv("D:\\Study\\Python\\Project 1 - Dataset(3).csv")

#Fetch data
n=10
weight=data["Weight lbs"]
height=data["Height inch"]
neck_cir=data["Neck circumference"]
chest_cir=data["Chest circumference"]
abdomen=data["Abdomen  circumference"]
bodyfat=data["bodyfat"]
df=pa.DataFrame(data=[weight,height,neck_cir,chest_cir,abdomen])
df=pa.DataFrame.transpose(df)
X_train, X_test, y_train, y_test = train_test_split(df,bodyfat,test_size=0.25)
#y_train
X_train_norm = normalize(X_train)
#np.shape(X_train)
y_test=np.array([y_test])
Y_test_norm=normalize(y_test)
#X_train_norm
X_test_norm=normalize(X_test)
#np.shape(X_train_norm)
y_train=np.array([y_train])
Y_train_norm=normalize(y_train)
#np.shape(y_train)
#Y_train_norm
weight_list=np.random.uniform(-1,1,(500,5,10))
weight_list
#np.shape(weight_list)
#np.shape(Y_train_norm)
npop=500

def crossover_mutation(chrom,fitindex):
    #crossover
    #print("Fitness index in 2+: ",fitindex)
    chrom1=[]
    chrom1=copy.copy(chrom)
    parent1=chrom1[fitindex]
    #str1=parent[0:c_point]
    #str2=parent[c_point:]
    ch_ild=[]
    for i in range(0,500):
        parent2=chrom1[i]
        c_point=np.random.random_integers(2,len(parent1)-1)
        temp_ch_ild=parent1[0:c_point]+parent2[c_point:]
        temp_ch_ild2=parent1[c_point:]+parent2[0:c_point]
        ch_ild.append(temp_ch_ild)
        ch_ild.append(temp_ch_ild2)
    #mutation  
    for i in range(0,1000):
        for j in range(0,25):
            mut_point=np.random.random_integers(2,len(parent1)-1)
            if (ch_ild[i][mut_point]=="0"):
                ch_ild[i][0:mut_point]+"1"+ch_ild[i][mut_point:]
            else:
                ch_ild[i][0:mut_point]+"0"+ch_ild[i][mut_point:]
    return ch_ild

def deseg_denorm(child):
    child_new=[]
    child_denormalised=[]
    child_array=[]
    child_array=copy.copy(child)
    temp_str=""
    for i in range(0,1000):
        child_new.append([child_array[i][k:k+10] for k in range(0, len(child_array[i]), 10)])
        for j in range(0,50):
            temp_str=child_new[i][j]
            decimal_x=int(temp_str,2)
            temp_var=float(decimal_x)/1000
            child_new[i][j]=temp_var
            child_denormalised.append((temp_var*4)-2)
    child_denormalised=np.reshape(child_denormalised,(1000,5,10))
    return child_denormalised

def fitnessvalue1000(X_train_norm,child_denorm):
    childdenorm_arr=[]
    childdenorm_arr=copy.copy(child_denorm)
    yhat1= [None] * 1000
    for i in range(0,1000):
            x1=np.dot(X_train_norm,childdenorm_arr[i])
            f1=1/(1+np.exp(-x1))
            #print("f1" ,f1)
            yhat1[i]=f1.sum(axis=1)
            
    fitnessarray1=[None] * 1000
    fitness_values=[]
    #Finding the parent
    for j in range(0,1000):
        sumfit=0
        for k in range(0,189):
            fitnessarray1[j]=np.power((yhat1[j][k]-Y_train_norm[0][k]),2)
            sumfit=sumfit+fitnessarray1[j]
        fitness_values.append(((1-sumfit)/189)*100)
    return fitness_values

#fitness value
yhat= [None] * 500
for i in range(0,500):
        x=np.dot(X_train_norm,weight_list[i])
        f=1/(1+np.exp(-x))
        yhat[i]=f.sum(axis=1)

#Finding the parent
fitnessarray=[None] * 500
for j in range(0,500):
    sumfit=0
    for k in range(0,189):
        fitnessarray[j]=np.power((yhat[j][k]-Y_train_norm[0][k]),2)
        sumfit=sumfit+fitnessarray[j]
    fitnessvalue=((1-sumfit)/189)*100
    if(j==0):
        maxfit=fitnessvalue
    #print("fitnessvalue ",fitnessvalue)
    elif(fitnessvalue>maxfit):
        #print("maxfit: ",maxfit)
        maxfit=fitnessvalue
        fitnessindex=j
parent=maxfit
#print("Fitness index1: ",fitnessindex)

#Normalise the weight matrix from 0 to 1
norm_weight_list=[None] * 500
for n in range(0,500):
    norm_weight_list[n]=((weight_list[n]+2)/4)
norm_weight_list=[x*1000 for x in norm_weight_list]
norm_weight_list=np.round(norm_weight_list).astype(int)

#Convert to binary
bin_val=[]
temp_list=copy.copy(norm_weight_list.tolist())
#type(temp_list)
#bin_values=[None] * 500
for k in range(0,500):
    for i in range(0,5):
        for j in range(0,10):  
            tempvar=bin(temp_list[k][i][j])[2:].zfill(10)
            bin_val.append(tempvar)
            #bin_values[k]=bin_val
bin_val=np.reshape(bin_val,(500,5,10))

#Chromosome
chrom=[]
newstr=""
for i in range(0,500):
    newstr=""
    for j in range(0,5):
        for k in range(0,10):
            newstr=""+newstr+bin_val[i][j][k]
    chrom.append(newstr)       

#Crossover and Mutation 
child=[]
child=crossover_mutation(chrom,fitnessindex)

child_prev=chrom
index_prev=fitnessindex

plotdata=[]
plotindex=[]
plotcount=0
for i in range(0,100):
    plotcount+=1
    #Desegment each chromosome into 10 bits,convert to decimal,divide by 1000,denormalise it
    child_denorm=[]
    child_denorm=deseg_denorm(child)
    #Find fitness_value for 1000 inputs
    fitnessvalues=[]
    fitnessvalues=fitnessvalue1000(X_train_norm,child_denorm)
    #Reduce 2nop to npop by finding 500 fittest people
    index=[]
    index.append(np.argsort(fitnessvalues))
    highest_fitness=[]
    new_childern=[]
    for i in range(500,1000):
        temp_ind=index[0][i]
        highest_fitness.append(fitnessvalues[temp_ind])
        new_childern.append(child[temp_ind])
    #saving highest fitness value as parent
    gr_index=index[0][len(index)-1]
    current_parent=highest_fitness[len(highest_fitness)-1]
    current_highest_index=len(highest_fitness)-1
    
    #Compare current parent with previous parent
    if(current_parent>parent):
        print("Difference in fitness value: ",current_parent-parent)
        print("Parent is: ",current_parent)
        parent=current_parent
        child_prev=new_childern
        index_prev=current_highest_index
        #Crossover mutation again
        child=[]
        child=crossover_mutation(new_childern,current_highest_index)
    else:
        print("Current parent value is less than or equal to previous parent",current_parent)
        child=[]
        child=crossover_mutation(child_prev,index_prev)
        
    plotdata.append(current_parent)
    plotindex.append(plotcount)

#Plot a scatter plot
py.scatter(plotindex,plotdata)
py.xlabel("Iterations")
py.ylabel("Fitness Value")
py.show()
        
#Calculate yhat for test set
ylasthat=[]
#for i in range(0,63):
for j in (0,10):
    z=np.dot(X_test_norm,child_denorm[current_highest_index])
    v=1/(1+np.exp(-z))
    ylasthat.append(v.sum(axis=1))
        
#3D Scatter plot
fig = py.figure()
ax = Axes3D(fig)
heightx=[]
weighty=[]
for i in range(0,len(X_test_norm)):
    heightx.append(X_test_norm[i][0])
    weighty.append(X_test_norm[i][1])
    
ax.scatter(heightx, weighty, ylasthat[0])
ax.scatter(heightx, weighty, Y_test_norm)
py.show()

#Error
sumfite=0
for i in range(0,len(Y_test_norm)):
    sumfite=sumfite+np.power((ylasthat[0][i]-Y_test_norm[0][i]),2)
    errorvalue=(sumfite)/63
print("Error value: ",errorvalue)