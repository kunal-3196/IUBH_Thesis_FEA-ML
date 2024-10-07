# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 23:43:59 2024

@author: LENOVO
"""

import pandas as pd
import numpy as np
import sqlite3 as sq
import sqlalchemy as sqlal
import statistics
import sympy as sp
# import boklehas as bk
import matplotlib.pyplot as plt
import csv  
import bokeh.plotting as bk
import datetime as dt
from dateutil import parser
from sympy import *
from sklearn import linear_model
from keras.layers import Dense
from keras import Sequential
import time
from keras import metrics as m1
from sklearn.datasets import make_blobs
 
# creating datasets X containing n_samples
# Y containing two classes

a,b=symbols('a b')

def SVM_Classifier(input_dataframe,Y_neural,hyperplane_at):
    
    maximum_deformation=max(input_dataframe["Total Deformation"])
    minimum_deformation=min(input_dataframe["Total Deformation"])
    
    # hyperplane_at=(maximum_deformation+minimum_deformation)/2
    if(hyperplane_at==0):
        hyperplane_at=np.mean(input_dataframe["Total Deformation"])
    # SV2_dist=[0]*len(input_dataframe)
    SV2_dist=[]
    SV1_dist=[]
    # SV1_dist=[0]*len(input_dataframe)
    # print(input_dataframe)
    for i in range(len(input_dataframe)):
        input_dataframe.iat[i,5]=maximum_deformation
        input_dataframe.iat[i,6]=maximum_deformation
        if(input_dataframe.iat[i,4]>=hyperplane_at):
            input_dataframe.iat[i,5]=(np.sqrt(np.abs(np.square(hyperplane_at) -np.square(input_dataframe.iat[i,4]))))
        else:
            input_dataframe.iat[i,6]=(np.sqrt(np.abs(np.square(hyperplane_at) -np.square(input_dataframe.iat[i,4]))))
    # print(input_dataframe)
    # print(input_dataframe["SV2_Dist"].drop)
    # print(hyperplane_at+min(SV2_dist))
    # print(hyperplane_at-min(SV1_dist))
    # print(SV1_dist)
    # SV2=hyperplane_at
    # SV2=hyperplane_at
    print("Hyperplane_at",hyperplane_at)
    # for i in range(len(input_dataframe)):
    #     if(input_dataframe.iat[1,5]==0):
    #         input_dataframe.iat[i,5]==max(input_dataframe["SV2_Dist"])
    #     # if(input_dataframe.iat[i,5]>0):
    #     #     SV2_dist.append(input_dataframe.iat[i,5])
    #     # if(input_dataframe.iat[1,6]==0):
    #     #     input_dataframe.iat[i,6]==max(input_dataframe["SV1_Dist"])
    #     if(input_dataframe.iat[i,6]>0):
    #         SV1_dist.append(input_dataframe.iat[i,6])
    # plt.scatter(input_dataframe["Node_No"],input_dataframe["Total Deformation"])

    
    # X,Y= make_blobs(n_samples=500, centers=2,
    #                   random_state=0, cluster_std=0.40)
    print("Final Deform is ",input_dataframe["Total Deformation"])
    
    X=list(np.array(input_dataframe["Node_No"]))
    # print(X[:,0])
    print("SV2 at",hyperplane_at+min(input_dataframe["SV2_Dist"]))
    print( "SV1 at",hyperplane_at+min(input_dataframe["SV1_Dist"]))
    # print(X)
    Y=list(np.array(input_dataframe["Total Deformation"]))
        # print(X[i])
        # Score_array[i][1]=Y[i]
        # Score_array[i,i]=[1,1]
    # Score_array_2 = np.vstack(Score_array)
    # Y=np.array([0.020515713,0.040564751,0.05,0.073787414,0.06072987,0.096226213,0.025001338,0.019416488,0.02500308,0.021625302,0.025005479,0.01627145,0.019431081,0.021013607,0.024996004,0.01743631,0.025005809,0.025003338,0.022253353,0.02278679,0.0250045,0.019416488,0.020930745,0.02500308,0.021297455,0.020422147,0.018364675,0.016463736,0.020969182,0.02238983,0.017677752,0.01551156,0.020687639,0.020144173,0.020009343,0.015392791,0.02106066,0.022180712,0.017444415,0.02071281,0.025003338,0.019308843,0.019437945,0.01169788,0.011000041,0.016033169,0.020373576,0.020279135,0.016041378,0.020556113,0.016881528,0.017797623,0.019710873,0.025003338,0.020876391,0.02500308,0.025005809,0.0250045,0.025001338,0.025,0.024997396,0.024996004,0.025005479,0.025005809,0.018479094,0.0250045,0.025001338,0.025,0.024997396,0.024996004,0.025005479,0.02500308,0.025003338,0.018867962,0.025005479,0.017063555,0.024996004,0.016090059,0.024997396,0.016198793,0.025,0.017373339,0.025001338,0.016315713,0.013630939,0.01171843,0.013589717,0.017008613,0.0250045,0.018867962,0.025005809,0.015658943,0.012837543,0.010785249,0.010000045,0.010762918,0.012793764,0.015612818,])
    
    Y1=[0]*len(Y)
    if (len(Y_neural)==0):
        for i in range(len(Y)):
            if Y[i]>=hyperplane_at:
             # if Y[i]>=hyperplane_at+min(SV2_dist):
                Y1[i]=1
              # if Y[i]<=hyperplane_at-min(SV1_dist):
                # Y1.append(0)
            else:
                Y1[i]=0
        SV2_pt=(input_dataframe[input_dataframe["SV2_Dist"]==min(input_dataframe["SV2_Dist"])].index[0])
        SV1_pt=(input_dataframe[input_dataframe["SV1_Dist"]==min(input_dataframe["SV1_Dist"])].index[0])
        print(Y1)
        # Y2=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        # print(X)
        # print(Y)
        # plotting scatters 
        plt.scatter(Y,X, c=Y1, s=10, cmap='spring');
        from sklearn import svm
        from sklearn.model_selection import cross_val_score
        Score_array=[]
        for i in range(len(X)):
            Score_array.append([X[i],Y[i]])
        svm_classifier = svm.SVC()
        svm_classifier.fit(Score_array, np.array(Y1))
        accuracy = svm_classifier.score(Score_array, np.array(Y1))
        print("Accuracy of SVM is",accuracy)
        scores = cross_val_score(svm_classifier, Score_array, Y1, cv=5)
        mean_score = scores.mean()
        print("Mean Score",mean_score)
        # plt.colorbar(label='Class Labels',bbox_to_anchor=(0, 1))
        plt.axvline(x=hyperplane_at+min(input_dataframe["SV2_Dist"]),color="green",linestyle="dotted",label="Support Vector 2")
        plt.axvline(x=hyperplane_at-min(input_dataframe["SV1_Dist"]),color='red',linestyle="dotted",label="Support Vector 1")
        # plt.axvline(x=input_dataframe.iat[SV2_pt,4],color="green",linestyle="dotted",label="Support Vector 2")
        # plt.axvline(x=input_dataframe.iat[SV1_pt,4],color="red",linestyle="dotted",label="Support Vector 1")    
        plt.axvline(x=hyperplane_at,color="black",linestyle="dashed",label="Hyperplane")
        plt.xlabel("Total Deformation per unit length of element(Strain)")
        plt.ylabel("Node Number")
        plt.title("SVM Classifier for Nodal failure")
        # plt.xticks(np.linspace(0.005, 0.025,20))
        plt.autoscale()
        # plt.scatter(Y,X, c=Y2, s=50, cmap='spring');
        # plt.legend()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0))
        plt.show()
    else:
        for i in range(len(Y)):
            if (Y[i]>=hyperplane_at or Y_neural[i]==1):
             # if Y[i]>=hyperplane_at+min(SV2_dist):
                Y1[i]=1
              # if Y[i]<=hyperplane_at-min(SV1_dist):
                # Y1.append(0)
            else:   
                Y1[i]=0
        # Y1=Y_neural
        SV2_pt=(input_dataframe[input_dataframe["SV2_Dist"]==min(input_dataframe["SV2_Dist"])].index[0])
        SV1_pt=(input_dataframe[input_dataframe["SV1_Dist"]==min(input_dataframe["SV1_Dist"])].index[0])
        print(Y1)
        # Y2=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        # print(X)
        # print(Y)
        # plotting scatters 
        plt.scatter(Y,X, c=Y1, s=10, cmap='spring');
        # plt.colorbar(label='Class Labels',bbox_to_anchor=(0, 1))
        plt.axvline(x=hyperplane_at+min(input_dataframe["SV2_Dist"]),color="green",linestyle="dotted",label="Support Vector 2")
        plt.axvline(x=hyperplane_at-min(input_dataframe["SV1_Dist"]),color='red',linestyle="dotted",label="Support Vector 1")
        # plt.axvline(x=input_dataframe.iat[SV2_pt,4],color="green",linestyle="dotted",label="Support Vector 2")
        # plt.axvline(x=input_dataframe.iat[SV1_pt,4],color="red",linestyle="dotted",label="Support Vector 1")    
        plt.axvline(x=hyperplane_at,color="black",linestyle="dashed",label="Hyperplane")
        plt.xlabel("Total Deformation per unit length of element(Strain)")
        plt.ylabel("Node Number")
        plt.title("SVM Classifier for Nodal failure post Neural Network Processing")
        # plt.xticks(np.linspace(0.005, 0.025,20))
        plt.autoscale()
        # plt.scatter(Y,X, c=Y2, s=50, cmap='spring');
        # plt.legend()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0))
        plt.show()
    return(hyperplane_at,Y1)
    # plt.show() 
    
def regression_function(input_dataframe,stress,new_node,new_stress):
    new_col=len(input_dataframe.axes[1])
    # print(new_col)
    input_dataframe.insert(new_col,"Stress",0)
    input_dataframe.insert(new_col+1,"Tangent Modulus",0)
    for i in range(len(input_dataframe)):
        strain=input_dataframe.iat[i,4]/1
        input_dataframe.iat[i,new_col]=stress
        input_dataframe.iat[i,new_col+1]=stress/strain/10**9
    # print(input_dataframe)
    # print(sum(input_dataframe["Tangent Modulus"])/98)
    X=input_dataframe[["Node_No","Stress"]]
    Y=input_dataframe[["Total Deformation"]]
    regression_function=linear_model.LinearRegression()
    regression_function.fit(X, Y)
    # plt.plot(regression_function)
    pred_TM=regression_function.predict([[2,4000000]])
    Nodal_Mean=np.mean(input_dataframe["Node_No"])
    Mean_Deformation=np.mean(input_dataframe["Tangent Modulus"])
    Sum_Node=0
    Sum_Cov=0
    # print(input_dataframe)
    for i in range(len(input_dataframe)):
        # print(input_dataframe.iat[i,0])
        Sum_Node+=(input_dataframe.iat[i,0])**2
        Sum_Cov+=(input_dataframe.iat[i,new_col+1])*(input_dataframe.iat[i,0])
    Cov=Sum_Cov/Sum_Node    
    print(Cov)
    eqn=(b-int(Mean_Deformation))-int(Cov)*(a-Nodal_Mean)
    print("The eqn of reg line ",str(eqn))
    input_dataframe.insert(new_col+2,"Reg_Tangent_Modulus",0)
    for i in range(len(input_dataframe)):
        input_dataframe.iat[i,new_col+2]=solve(eqn.subs({a:i}))[0]
    # print(input_dataframe)
    input_dataframe.insert(new_col+3,"Error_Tangent_Modulus",0)
    for i in range(len(input_dataframe)):
        input_dataframe.iat[i,new_col+3]=input_dataframe.iat[i,new_col+1]-input_dataframe.iat[i,new_col+2]
    # print(input_dataframe)
    Mean_Error=np.mean(input_dataframe["Error_Tangent_Modulus"])
    Sum_Cov_error=0
    # print(input_dataframe)
    for i in range(len(input_dataframe)):
        # print(input_dataframe.iat[i,0])
        Sum_Node+=(input_dataframe.iat[i,0])**2
        Sum_Cov_error+=(input_dataframe.iat[i,new_col+3])*(input_dataframe.iat[i,0])
    Cov_error=Sum_Cov_error/Sum_Node    
    print(Cov_error)
    eqn_error=(b-int(Mean_Error))-(int(Cov_error)*(a-Nodal_Mean))
    print(str(eqn_error))
    input_dataframe.insert(new_col+4,"Reg_Error_Tangent_Modulus",0)
    for i in range(len(input_dataframe)):
        input_dataframe.iat[i,new_col+4]=solve(eqn_error.subs({a:i}))[0]
    print(input_dataframe)
    input_dataframe.to_csv("C:\\MISC\\IU_downloads\\Thesis\\crankshaft_output_reg_error.csv")
    input_dataframe.insert(new_col+5,"Reg_with_Error_Tangent_Modulus",0)
    for i in range(len(input_dataframe)):
        if(input_dataframe.iat[i,new_col+2]+(np.sqrt(len(input_dataframe["Node_No"]))*input_dataframe.iat[i,new_col+4])<0):
            input_dataframe.iat[i,new_col+5]=0
        else:
            input_dataframe.iat[i,new_col+5]=input_dataframe.iat[i,new_col+2]+(np.sqrt(len(input_dataframe["Node_No"]))*input_dataframe.iat[i,new_col+4])
    print(input_dataframe)
    input_dataframe.insert(new_col+6,"Predicted Tangent Modulus",0)
    for i in range(len(input_dataframe)):
        input_dataframe.iat[i,new_col+6]=round(input_dataframe.iat[i,new_col+2]+(np.sqrt(len(input_dataframe["Node_No"]))*input_dataframe.iat[i,new_col+4]),4)
    print(input_dataframe)
    # print(f"Coefficents of model: {regression_function.coef_}")
    # Strain= target variable, Higher tangent mod=safer structure  
    plt.plot(input_dataframe["Node_No"],input_dataframe["Tangent Modulus"], color='orange',label="Curve of Tangent Modulus")
    # plt.plot(input_dataframe["Node_No"],input_dataframe["Total Deformation"])
    # plt.plot(input_dataframe["Node_No"],input_dataframe["Reg_Tangent_Modulus"])
    plt.plot(input_dataframe["Node_No"],input_dataframe["Reg_with_Error_Tangent_Modulus"], color='blue',label="Line of Convergence") 
    # print(predictied_strain/(10**9))
    for i in range(len(new_stress)):
        start_node=list(input_dataframe["Node_No"])[0]-1
        end_node=list(input_dataframe["Node_No"])[len(list(input_dataframe["Node_No"]))-1]
        print("Bool st end node",new_node[i]>=start_node & new_node[i]<=end_node)
        # print("Bool st end node", new_node[i]<=end_node)
        print("Input node:",new_node[i])
        print("End node:",end_node)
        if(new_node[i]>=start_node and new_node[i]<=end_node):
            new_deformation=new_stress[i]/(input_dataframe.iat[int(new_node[i]-start_node),8]*10**9)
            input_dataframe.iat[new_node[i]-start_node,4]=new_deformation
            input_dataframe.iat[new_node[i]-start_node,new_col+6]=input_dataframe.iat[int(new_node[i]- start_node),8]
        # print("New Deform",new_deformation)
        
            plt.scatter(new_node[i],input_dataframe.iat[int(new_node[i]-start_node),new_col+6],color='red')
            print("New deformation",new_deformation)
    # new_tangent_mod=stress/(new_deformation*10**9)
    # print("New Tangent Mod",new_tangent_mod)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.xlabel("Node Number")
    plt.ylabel("Tangent Modulus")
    plt.title("Regression Plot for Convergence of Tangent Modulus")
    plt.show()
    # plt.plot(input_dataframe["Tangent Modulus"],input_dataframe["Total Deformation"])
    # plt.show()
    return(input_dataframe)
    
def CNN_pred(input_data_frame,Y_bool):
    data = np.asarray(input_data_frame["Predicted Tangent Modulus"]).astype(np.float32)
    data.shape = len(input_data_frame), 1
    print("This is TMod",input_data_frame["Tangent Modulus"])
    # labels = np.array(input_data_frame["Total Deformation"])
    labels = np.array(Y_bool)
    print(labels[:50])
    model = Sequential([
    Dense(units=20, input_shape=(1,), activation='relu'),
    Dense(units=2, activation='softmax'),])
    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    # loss='binary_crossentropy',
    metrics=['accuracy'])
    # metrics=['accuracy'])
    model.summary()
    model.fit(x=data, y=labels, epochs=2, verbose=1)
    m2=m1.CategoricalCrossentropy()
    m2.update_state([[0, 0, 1], [0, 1, 0]],[[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    print("This is Keras Final Accuracy", m2.result()   )
    # data_test=np.array([0.106361,0.107996,0.087131, 0.088440, 0.073452])
    # data_test=np.array([0.020515713,0.040564751,0.05,0.073787414,0.06072987,0.096226213,0.025001338,0.019416488,0.02500308,0.021625302,0.025005479,0.01627145,0.019431081,0.021013607,0.024996004,0.01743631,0.025005809,0.025003338,0.022253353,0.02278679,0.0250045,0.019416488,0.020930745,0.02500308,0.021297455,0.020422147,0.018364675,0.016463736,0.020969182,0.02238983,0.017677752,0.01551156,0.020687639,0.020144173,0.020009343,0.015392791,0.02106066,0.022180712,0.017444415,0.02071281,0.025003338,0.019308843,0.019437945,0.01169788,0.011000041,0.016033169,0.020373576,0.020279135,0.016041378,0.020556113,0.016881528,0.017797623,0.019710873,0.025003338,0.020876391,0.02500308,0.025005809,0.0250045,0.025001338,0.025,0.024997396,0.024996004,0.025005479,0.025005809,0.018479094,0.0250045,0.025001338,0.025,0.024997396,0.024996004,0.025005479,0.02500308,0.025003338,0.018867962,0.025005479,0.017063555,0.024996004,0.016090059,0.024997396,0.016198793,0.025,0.017373339,0.025001338,0.016315713,0.013630939,0.01171843,0.013589717,0.017008613,0.0250045,0.018867962,0.025005809,0.015658943,0.012837543,0.010785249,0.010000045,0.010762918,0.012793764,0.015612818,])
    data_test=np.array(input_data_frame["Total Deformation"]).astype(np.float32)
    label_out=model.predict(data_test)
    print(label_out)
    predicted_SVM_label=[0]*len(input_data_frame)
    label_diff=[]
    for i in range(len(label_out)):
        label_diff.append(label_out[i][1]-label_out[i][0])        
    for i in range(len(label_out)):   
        # if(label_out[i][0]>=label_out[i][1]-np.mean(np.array(label_diff))):
        if(label_out[i][0]>=label_out[i][1]):
            predicted_SVM_label[i]=0
        else:
            predicted_SVM_label[i]=1
            
    print(predicted_SVM_label)
    return(predicted_SVM_label)                

# start = time.time()    
# extrapolated_FEA_result=pd.read_csv("C:\\MISC\IU_downloads\\Thesis\\extrapolated_frame_output.csv")
# extrapolated_FEA_result.insert(5,"SV2_Dist",0)
# extrapolated_FEA_result.insert(6,"SV1_Dist",0)
# y_SVM=SVM_Classifier(extrapolated_FEA_result,[],0)
# # regression_function(extrapolated_FEA_result,2178283,4,10000000)
# reg_dataframe=regression_function(extrapolated_FEA_result,2178283,[22,35,44,38,12],[4000000,4800000,6000000,5000000,4000000])
# # print("This is reg_deform",reg_dataframe["Total Deformation"])
# # data_test=np.array([0.020515713,0.040564751,0.05,0.073787414,0.06072987,0.096226213,0.025001338,0.019416488,0.02500308,0.021625302,0.025005479,0.01627145,0.019431081,0.021013607,0.024996004,0.01743631,0.025005809,0.025003338,0.022253353,0.02278679,0.0250045,0.019416488,0.020930745,0.02500308,0.021297455,0.020422147,0.018364675,0.016463736,0.020969182,0.02238983,0.017677752,0.01551156,0.020687639,0.020144173,0.020009343,0.015392791,0.02106066,0.022180712,0.017444415,0.02071281,0.025003338,0.019308843,0.019437945,0.01169788,0.011000041,0.016033169,0.020373576,0.020279135,0.016041378,0.020556113,0.016881528,0.017797623,0.019710873,0.025003338,0.020876391,0.02500308,0.025005809,0.0250045,0.025001338,0.025,0.024997396,0.024996004,0.025005479,0.025005809,0.018479094,0.0250045,0.025001338,0.025,0.024997396,0.024996004,0.025005479,0.02500308,0.025003338,0.018867962,0.025005479,0.017063555,0.024996004,0.016090059,0.024997396,0.016198793,0.025,0.017373339,0.025001338,0.016315713,0.013630939,0.01171843,0.013589717,0.017008613,0.0250045,0.018867962,0.025005809,0.015658943,0.012837543,0.010785249,0.010000045,0.010762918,0.012793764,0.015612818,])
# Y_neural=CNN_pred(reg_dataframe,y_SVM[1])
# y_SVM2=SVM_Classifier(reg_dataframe, Y_neural,y_SVM[0])
# end = time.time()
# print("The time consumed is",end-start)
# # SVM_Classifier()
    
start = time.time()
nodal_disp_crankshaft=pd.read_csv("C:\\MISC\IU_downloads\\Thesis\\nodal_disp_crankshaft_v1.2.csv")
nodal_disp_crankshaft.insert(5,"SV2_Dist",0)
nodal_disp_crankshaft.insert(6,"SV1_Dist",0)
# # y_SVM=SVM_Classifier(nodal_disp_crankshaft,[],0)
y_SVM=SVM_Classifier(nodal_disp_crankshaft,[],0)
# regression_function(extrapolated_FEA_result,2178283,4,10000000)
reg_dataframe=regression_function(nodal_disp_crankshaft,(18.628977853467863*(10**9)),[12082],[(98.6289*(10**9))])
# # reg_dataframe=regression_function(nodal_disp_crankshaft[:10000],(18.628977853467863*(10**9)),[15,289,12082,20085,38922],[(1980.6289*(10**9)),(2500.6289*(10**9)),(1482.6289*(10**9)),(2198.6289*(10**9)),(1198.6289*(10**9))])
# # # print("This is reg_deform",reg_dataframe["Total Deformation"])
# # # data_test=np.array([0.020515713,0.040564751,0.05,0.073787414,0.06072987,0.096226213,0.025001338,0.019416488,0.02500308,0.021625302,0.025005479,0.01627145,0.019431081,0.021013607,0.024996004,0.01743631,0.025005809,0.025003338,0.022253353,0.02278679,0.0250045,0.019416488,0.020930745,0.02500308,0.021297455,0.020422147,0.018364675,0.016463736,0.020969182,0.02238983,0.017677752,0.01551156,0.020687639,0.020144173,0.020009343,0.015392791,0.02106066,0.022180712,0.017444415,0.02071281,0.025003338,0.019308843,0.019437945,0.01169788,0.011000041,0.016033169,0.020373576,0.020279135,0.016041378,0.020556113,0.016881528,0.017797623,0.019710873,0.025003338,0.020876391,0.02500308,0.025005809,0.0250045,0.025001338,0.025,0.024997396,0.024996004,0.025005479,0.025005809,0.018479094,0.0250045,0.025001338,0.025,0.024997396,0.024996004,0.025005479,0.02500308,0.025003338,0.018867962,0.025005479,0.017063555,0.024996004,0.016090059,0.024997396,0.016198793,0.025,0.017373339,0.025001338,0.016315713,0.013630939,0.01171843,0.013589717,0.017008613,0.0250045,0.018867962,0.025005809,0.015658943,0.012837543,0.010785249,0.010000045,0.010762918,0.012793764,0.015612818,])
Y_neural=CNN_pred(reg_dataframe,y_SVM[1])
y_SVM2=SVM_Classifier(reg_dataframe, Y_neural,y_SVM[0])
end = time.time()
print("The time consumed is",end-start)
    
    