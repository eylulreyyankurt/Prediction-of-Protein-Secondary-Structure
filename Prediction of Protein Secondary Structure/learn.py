import math
from math import exp
import random
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

import numpy as np
protein_data = np.load(path + "/cb513_processed_protein.npy")
for i in range(len(protein_data)):
    protein_data[i]=np.array(protein_data[i])

#将数据集随机分成测试集和训练集
from sklearn.model_selection import train_test_split
protein_data_train, protein_data_test = train_test_split(protein_data, test_size=0.2)

para=list(np.load(path +"/parameters/10_parameters.npy"))

#设定参数
#其他
m_RFrameWidth=para[0] #窗宽
seed=para[1] #随机种子
m_HLUnitNum=para[2] #隐含神经元个数
count=[0]


#隐含层
m_YJieV=para[3] #截距参数
m_YXingV=para[4] #形状参数
m_YShouV=para[5] #收敛参数


#输出层
m_OJieV=para[6] #截距参数
m_OXingV=para[7] #形状参数
m_OShouV=para[8] #收敛参数




#设定初始隐含层权值矩阵（不用传进函数里）
import random
random.seed(seed)
m_WeightHidden=[]
for j in range(m_HLUnitNum):
    n_WeightHidden=[]
    for i in range(22):
        n_WeightHidden.append(np.array([random.uniform(-0.3,0.3) for i in range((2*m_RFrameWidth+1))]))
    n_WeightHidden=np.array(n_WeightHidden)  
    m_WeightHidden.append(n_WeightHidden)
m_WeightHidden=np.array(m_WeightHidden)

#设定初始输出层权值矩阵（不用传进函数里）
import random
random.seed(seed)
m_WeightHelix=[random.uniform(-0.3,0.3) for i in range(m_HLUnitNum)]
m_WeightSheet=[random.uniform(-0.3,0.3) for i in range(m_HLUnitNum)]
m_WeightCoil=[random.uniform(-0.3,0.3) for i in range(m_HLUnitNum)]



#定义函数：学习一个蛋白质(训练集中的蛋白质)
def learn_ami(NowProtein_id):

    '''        
    #传入各层矩阵
    m_WeightHidden=np.load("/output_weight/weight_read/m_WeightHidden.npy")
    m_WeightHelix=np.load("/output_weight/weight_read/m_WeightHelix.npy")
    m_WeightCoil=np.load("/output_weight/weight_read/m_WeightCoil.npy")
    m_WeightSheet=np.load("/output_weight/weight_read/m_WeightSheet.npy")
    '''        
       
    m_2DArray=protein_data_train[NowProtein_id][:,21] #该蛋白质的2级结构序列
    m_AminoArray=protein_data_train[NowProtein_id][:,:21] #该蛋白质的氨基酸序列
    m_AminoArray=np.concatenate((m_AminoArray,np.array([[0]]*len(m_AminoArray))),axis=1)   #在每个氨基酸独热码后加一项，表明reading frame的该位置没有氨基酸
    m_AminoNum=len(protein_data_train[NowProtein_id])
    
            
    #生成氨基酸读取顺序列表（shuffle生成）
    random.seed(seed)
    m_ChooseOrder=[int(i) for i in range(m_AminoNum)]
    random.shuffle(m_ChooseOrder)
    
    #对每一个氨基酸位点进行学习
    
    TempWholeNum=22*(2*m_RFrameWidth+1)
    HiddenNet=[0 for i in range(m_HLUnitNum)]
    HiddenResult=[0 for i in range(m_HLUnitNum)]
    DoushuForHiddenUnit=[0 for i in range(m_HLUnitNum)]
    
    for k in range(m_AminoNum):
        NowCentreAmino=m_ChooseOrder[k]
        
        #读取2级结构
        if m_2DArray[NowCentreAmino]=="H":
            TagetOfHelix=1
            TagetOfSheet=0
            TagetOfCoil=0
            TagetResultNumber=1
        elif m_2DArray[NowCentreAmino]=="E":
            TagetOfHelix=0
            TagetOfSheet=1
            TagetOfCoil=0
            TagetResultNumber=2
        else:
            TagetOfHelix=0
            TagetOfSheet=0
            TagetOfCoil=1
            TagetResultNumber=3
            
        #读取氨基酸(生成阅读框)
        m_ReadingFrame=[]
        for i in range(-m_RFrameWidth,m_RFrameWidth+1):
            if m_ChooseOrder[k]+i>=0 and m_ChooseOrder[k]+i<m_AminoNum:
                m_ReadingFrame.append(m_AminoArray[m_ChooseOrder[k]+i])
            else:
                m_ReadingFrame.append(np.array([0 for i in range(21)]+[1]))
        m_ReadingFrame=np.array(m_ReadingFrame) 
        m_ReadingFrame=np.transpose(m_ReadingFrame) #将reading_frame进行转置处理！！！！
    
        
        #在未达到预测结果前不断调整权值
        while(1):
            NetjForHelix=0
            NetjForSheet=0
            NetjForCoil=0
            
            #对隐含层求值
            for m in range(m_HLUnitNum):
                
                #隐含层m节点的net值
                for i in range(TempWholeNum):
                    HiddenNet[m]+=float(m_ReadingFrame.reshape(-1)[i])*float(m_WeightHidden[m].reshape(-1)[i])
    			
                #隐含层m节点的输出值
                HiddenResult[m]=float(1/(1+exp((-1)*(HiddenNet[m]+m_YJieV)/m_YXingV)))
    
                
                
    
    		#对输出层三种二级结构类型求netj
            for i in range(m_HLUnitNum):
                NetjForHelix+=HiddenResult[i]*m_WeightHelix[i]
                NetjForSheet+=HiddenResult[i]*m_WeightSheet[i]
                NetjForCoil+=HiddenResult[i]*m_WeightCoil[i]
    				
    		#求输出层三种二级结构对应的输出层神经元的输出
            OutputOfHelix=float(1/(1+exp((-1)*(NetjForHelix+m_OJieV)/m_OXingV)))
            OutputOfSheet=float(1/(1+exp((-1)*(NetjForSheet+m_OJieV)/m_OXingV)))
            OutputOfCoil=float(1/(1+exp((-1)*(NetjForCoil+m_OJieV)/m_OXingV)))
    		
            #寻找三个输出值中最大的一个
            if OutputOfHelix>=OutputOfSheet and OutputOfHelix>=OutputOfCoil:
                ResultNumber=1
            elif OutputOfSheet>OutputOfHelix and OutputOfSheet>=OutputOfCoil:
                ResultNumber=2
            else:
                ResultNumber=3
    				
    		#如果预测结果同实际结果相同就退出,否则修改权重继续运算
            if ResultNumber==TagetResultNumber:
                break
            else:
                #求导数
                daoshuForHelix=float(exp((-1)*(NetjForHelix+m_OJieV)/m_OXingV)/(m_OXingV*(1+exp((-1)*(NetjForHelix+m_OJieV)/m_OXingV))*(1+exp((-1)*(NetjForHelix+m_OJieV)/m_OXingV))))
                daoshuForSheet=float(exp((-1)*(NetjForSheet+m_OJieV)/m_OXingV)/(m_OXingV*(1+exp((-1)*(NetjForSheet+m_OJieV)/m_OXingV))*(1+exp((-1)*(NetjForSheet+m_OJieV)/m_OXingV))))
                daoshuForCoil=float(exp((-1)*(NetjForCoil+m_OJieV)/m_OXingV)/(m_OXingV*(1+exp((-1)*(NetjForCoil+m_OJieV)/m_OXingV))*(1+exp((-1)*(NetjForCoil+m_OJieV)/m_OXingV))))
                
                DeltaHelix=(TagetOfHelix-OutputOfHelix)*daoshuForHelix
                DeltaSheet=(TagetOfSheet-OutputOfSheet)*daoshuForSheet
                DeltaCoil=(TagetOfCoil-OutputOfCoil)*daoshuForCoil
                
                ReverseWeightSumHiddenUnit=[m_WeightHelix[i]*DeltaHelix+m_WeightSheet[i]*DeltaSheet+m_WeightCoil[i]*DeltaCoil for i in range(m_HLUnitNum)]
    										
    			#对输出层修正权值
                for i in range(m_HLUnitNum):
                    m_WeightHelix[i]+=m_OShouV*DeltaHelix*HiddenResult[i]
                    m_WeightSheet[i]+=m_OShouV*DeltaSheet*HiddenResult[i]
                    m_WeightCoil[i]+=m_OShouV*DeltaCoil*HiddenResult[i]
    					
    			#对隐含层修正权值
                for i in range(m_HLUnitNum):
                    DoushuForHiddenUnit[i]=float(exp((-1)*(HiddenNet[i]+m_YJieV)/m_YXingV)/(m_YXingV*(1+exp((-1)*(HiddenNet[i]+m_YJieV)/m_YXingV))*(1+exp((-1)*(HiddenNet[i]+m_YJieV)/m_YXingV))))
    			    
                    m_weight_temp=m_WeightHidden[i].reshape(-1)
                    for m in range(TempWholeNum):
                        m_weight_temp[m]+=m_YShouV*ReverseWeightSumHiddenUnit[i]*DoushuForHiddenUnit[i]*float(m_ReadingFrame.reshape(-1)[m])
                    
                    m_WeightHidden[i]=m_weight_temp.reshape(22,(2*m_RFrameWidth+1))
                
                #每一轮都将权值矩阵存入weight_folder中
                #隐含矩阵的数据保存
                for i in range(len(m_WeightHidden)):
                    filename=path +"/output_weight/weight_cat/m_WeightHidden["+str(i)+"].txt"
                    np.savetxt(filename,m_WeightHidden[i])
                np.save(path +"/output_weight/weight_read/m_WeightHidden.npy", m_WeightHidden)
                
                #输出层的数据保存
                np.savetxt(path +"/output_weight/weight_cat/m_WeightHelix.txt",m_WeightHelix)
                np.save(path +"/output_weight/weight_read/m_WeightHelix.npy",m_WeightHelix)

                np.savetxt(path +"/output_weight/weight_cat/m_WeightSheet.txt",m_WeightSheet)
                np.save(path +"/output_weight/weight_read/m_WeightSheet.npy",m_WeightSheet)
                
                np.savetxt(path +"/output_weight/weight_cat/m_WeightCoil.txt",m_WeightCoil)
                np.save(path +"/output_weight/weight_read/m_WeightCoil.npy",m_WeightCoil)


def learn_protein(num):   
    
    #生成蛋白质读取顺序列表（shuffle生成）
    random.seed(seed)
    protein_r_order=[int(i) for i in range(len(protein_data_train))]
    random.shuffle(protein_r_order)
    
    #学习蛋白质
    for K in range(0,num):
        NowProtein_id=protein_r_order[K] #当前读取的蛋白质id号
        learn_ami(NowProtein_id)
        


