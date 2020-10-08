import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path)
import learn
import numpy as np
from math import exp

#请输入预测蛋白的id号：id=?
def predict(id):
    m_AminoNum=len(learn.protein_data_test[id])
    m_2DArray=learn.protein_data_test[id][:,21] #该蛋白质的2级结构序列
    m_AminoArray=learn.protein_data_test[id][:,:21] #该蛋白质的氨基酸序列
    m_AminoArray=np.concatenate((m_AminoArray,np.array([[0]]*len(m_AminoArray))),axis=1)   #在每个氨基酸独热码后加一项，表明reading frame的该位置没有氨基酸
    
    m_RFrameWidth=learn.m_RFrameWidth
    m_HLUnitNum=learn.m_HLUnitNum
    TempWholeNum=22*(2*m_RFrameWidth+1)
    m_YJieV=learn.m_YJieV
    m_YXingV=learn.m_YXingV
    m_YShouV=learn.m_YShouV
    
    m_OJieV=learn.m_OJieV
    m_OXingV=learn.m_OXingV
    m_OShouV=learn.m_OShouV
    
    #传入各层矩阵
    m_WeightHidden=np.load(path + "/output_weight/weight_read/m_WeightHidden.npy")
    m_WeightHelix=np.load(path + "/output_weight/weight_read/m_WeightHelix.npy")
    m_WeightCoil=np.load(path + "/output_weight/weight_read/m_WeightCoil.npy")
    m_WeightSheet=np.load(path + "/output_weight/weight_read/m_WeightSheet.npy")
    
    
    m_2DArrayResult=[] #创建输出序列
    
    for k in range(m_AminoNum):
        m_ReadingFrame=[]
        for i in range(-m_RFrameWidth,m_RFrameWidth+1):
            if k+i>=0 and k+i<m_AminoNum:
                m_ReadingFrame.append(m_AminoArray[k+i])
            else:
                m_ReadingFrame.append(np.array([0 for i in range(21)]+[1]))
        m_ReadingFrame=np.array(m_ReadingFrame) 
        m_ReadingFrame=np.transpose(m_ReadingFrame) #将reading_frame进行转置处理！！！！
        
        NetjForHelix=0
        NetjForSheet=0
        NetjForCoil=0
            
        HiddenNet=[0 for i in range(m_HLUnitNum)]
        HiddenResult=[0 for i in range(m_HLUnitNum)]
        
        #求隐含层各单元的输出		    
        for m in range(m_HLUnitNum):
            for i in range(TempWholeNum):
                HiddenNet[m]+=float(m_ReadingFrame.reshape(-1)[i])*m_WeightHidden[m].reshape(-1)[i]
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
            m_2DArrayResult.append("H")
        elif OutputOfSheet>OutputOfHelix and OutputOfSheet>=OutputOfCoil:
            m_2DArrayResult.append("E")
        else:
            m_2DArrayResult.append("C")			
    
    ###print(m_2DArrayResult)
    ###print(m_2DArray)       
    m_2DArray=[str(i) for i in m_2DArray]
    accur_num=[i for i in range(len(m_2DArrayResult)) if m_2DArrayResult[i]==m_2DArray[i]]
    accuracy=len(accur_num)/len(m_2DArray)

    #保存准确度
    back=open(path + "/predict_output/accuracy.txt","w")
    back.write("accuracy: "+str(accuracy))
    back.close

    #保存预测序列
    back_1=open(path + "/predict_output/predict_seq.txt","w")
    s="original_seq: "+"".join(m_2DArray)+"\n"+"predict_seq: "+"".join(m_2DArrayResult)
    back_1.write(s)
    back_1.close
    
    
