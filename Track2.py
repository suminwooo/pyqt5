import pywt
import warnings
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mode
warnings.filterwarnings('ignore')

########################## Definition ###############################

def extract_wave2(Denoise_Data):
    
    Denoise_Data = np.array(Denoise_Data)

    #### 0이 연달아 나올때 0 한개만 남기고 제거
    for i in range(len(Denoise_Data)):
        if Denoise_Data[0] != 0:
            i=0
            break
        else:
            if Denoise_Data[i+1] - Denoise_Data[i] != 0:
                break    
    front_zero_eliminate = Denoise_Data[i:]

    
    #### 데이터 안에서 연속되지 않은 wave를 구해준다.
    num=0
    LIST=[]
    while True:
        num = num+1
        
        ### 길이가 3이상인 것에 대해서만 wave를 정의하겠다.
        if len(front_zero_eliminate)>2:
            pass
        else:
            break
            
        ### wave 추출
        for idx in range(len(front_zero_eliminate)-1):
            
            ## 0이 연속 두번 나오면 0 사이에서 데이터를 잘라 앞쪽부분을 wave로 정의하고 for문을 마친다.
            if ((front_zero_eliminate[idx] == 0) &
                (front_zero_eliminate[idx+1] == 0)):

                end = idx+1
                
                front_zero_eliminate=front_zero_eliminate[end:]
                globals()['result{}'.format(num)] = front_zero_eliminate

                LIST.append(globals()['result{}'.format(num)])
                break
            
            ## 마지막 인덱스까지 0이 연속으로 두번 나오지 않으면, 마지막 인덱스+1까지 wave로 정의하고 for문을 마친다.
            else:
                if idx == len(front_zero_eliminate)-2:
                    end = idx + 1
                    front_zero_eliminate=front_zero_eliminate[end:]
                    result_end = front_zero_eliminate
                    LIST.append(result_end)
                    break
                # 0이 연속 두번 나오지 않거나 (끝-1)이 아니면 다시 for문으로 들어간다.
                else:
                    pass
 
        ### 0이 연달아 나오는 index 구하기
        for i in range(len(front_zero_eliminate)-1):
            if front_zero_eliminate[i+1] - front_zero_eliminate[i] != 0:
                break
    
        ### 마지막 인덱스까지 0이 연달아 나오지 않는다면 while문 종료. 아니면     
        if i==(len(front_zero_eliminate)-2):
            break
            
        ### 0이 연달아 나올때 0 한개만 남기고 제거
        else:
            front_zero_eliminate = front_zero_eliminate[i:]
        
        
    #### wave를 저장한 LIST의 크기가 1보다 크면 최대값이 가장 큰 wave를 출력
    #### LIST 안에 원소가 1개 밖에 없다면 그냥 그 값 출력
    if len(LIST)>1:
        vs=[]
        for element in LIST:
            vs.append(np.max(np.abs(element)))
        order=np.argmax(vs)
        result = globals()['result{}'.format(order+1)]

    else:
        result = LIST[0]

        
    if len(Denoise_Data) == len(result):
        print('Max_Data', result)
        
    return result



def ZeroCrossing(arr):
    k=0
    for i in range(len(arr)-1):
        if arr[i]*arr[i+1]<0:
            k += 1
    return k



def SlopeSignChange(arr):
    k=[]
    for i in range(len(arr)-1):
        if (arr[i+1]-arr[i])<0:
            k.append(-1)
        elif (arr[i+1]-arr[i])>0:
            k.append(1)
    h=0 
    for j in range(len(k)-1):
        if (k[j]+k[j+1])==0:
            h += 1
    return h

def Range(arr):
    Max_abs=np.abs(np.max(arr))
    Min_abs=np.abs(np.min(arr))
    if Max_abs > Min_abs:
        R = Min_abs / Max_abs
    else:
        R = Max_abs / Min_abs
        
    return R



class Track2_DataAnalysis:

    def __init__(self,PATH, file_name):
        self.PATH = PATH
        self.file_name = file_name
 
    def LoadData(self):      
        arr = np.array(pd.read_csv(self.PATH+'/'+self.file_name , header=None, low_memory=False))

        if len(arr[1,:]) == 201:
            arr = arr
            
        else:
            del_row = np.delete(arr, 0, axis=0)
            del_col = np.delete(del_row, [0,1,2,4,5,6], axis=1)
            arr = del_col

        return arr
    

    def Preprocess(self,arr):
        
        type1 = pd.DataFrame(arr).rename(columns={0:'pulse_topology'})

        # 데이터 전처리
        type1_clear = type1.iloc[:,1:]

        type1_clear = type1_clear - mode(type1_clear.iloc[:,1:],axis=1)[0]
        type1_1 = type1_clear.copy()

        max_value = type1_clear.iloc[:,1:].max(axis=1)
        for idx in range(1, 1 + len(type1_1.iloc[0,1:])):
            type1_1.iloc[:,idx] = type1_1.iloc[:,idx] / max_value


        # 입력 데이터 정의
        Denoise_Data=pd.DataFrame(np.zeros([len(type1_1),200]))

        # 데이터 분해
        for idx in range(len(type1_1)):

            cA33, cD33, cD32, cD31 = pywt.wavedec(type1_1.iloc[idx,1:], 'bior1.3', level=3, axis=-1)

            mcA3=pywt.threshold(cA33, np.std(cA33), mode='soft')
            mcD1=pywt.threshold(cD31, np.std(cD31), mode='soft')
            mcD2=pywt.threshold(cD32, np.std(cD32), mode='soft')
            mcD3=pywt.threshold(cD33, np.std(cD33), mode='soft')

            Denoise_Data.iloc[idx,:] = pywt.waverec([mcA3, mcD3, mcD2, mcD1], 'bior1.3', mode= 'symmetric', axis=-1) 
            
        return Denoise_Data
    

    def Feature_Extract(self, Denoise_Data):

        result = pd.DataFrame(range(len(Denoise_Data))).rename(columns={0:'index'})
        
        Denoise_Data_idx=Denoise_Data.reset_index()
        
        # 데이터 특징 추출
        for idx in range(len(Denoise_Data_idx)):

            cA11, cD11 = pywt.dwt(Denoise_Data_idx.iloc[idx,1:], 'bior1.3')
            cA22, cD22, _ = pywt.wavedec(Denoise_Data_idx.iloc[idx,1:], 'bior1.3', level=2, axis=-1)
            cA33, cD33, _, _ = pywt.wavedec(Denoise_Data_idx.iloc[idx,1:], 'bior1.3', level=3, axis=-1)

            result.loc[idx, 'A1_Range'] = Range(cA11)
            result.loc[idx, 'A2_Range'] = Range(cA22)
            result.loc[idx, 'A3_Range'] = Range(cA33)
            result.loc[idx, 'D1_Range'] = Range(cD11)
            result.loc[idx, 'D2_Range'] = Range(cD22)
            result.loc[idx, 'D3_Range'] = Range(cD33)

            result.loc[idx, 'A1_STD'] = np.std(cA11)
            result.loc[idx, 'D1_STD'] = np.std(cD11)
            result.loc[idx, 'D2_STD'] = np.std(cD22)
            result.loc[idx, 'D3_STD'] = np.std(cD33)

            result.loc[idx, 'D1_MAV'] = np.mean(abs(cD11))
            result.loc[idx, 'D2_MAV'] = np.mean(abs(cD22))
            result.loc[idx, 'D3_MAV'] = np.mean(abs(cD33))

            result.loc[idx, 'A1_skew'] = stats.skew(cA11)
            result.loc[idx, 'A2_skew'] = stats.skew(cA22)
            result.loc[idx, 'A3_skew'] = stats.skew(cA33)
            result.loc[idx, 'D1_skew'] = stats.skew(cD11)
            result.loc[idx, 'D2_skew'] = stats.skew(cD22)
            result.loc[idx, 'D3_skew'] = stats.skew(cD33)

            result.loc[idx, 'A1_KURT'] = stats.kurtosis(cA11)
            result.loc[idx, 'A2_KURT'] = stats.kurtosis(cA22)
            result.loc[idx, 'A3_KURT'] = stats.kurtosis(cA33)
            result.loc[idx, 'D1_KURT'] = stats.kurtosis(cD11)
            result.loc[idx, 'D2_KURT'] = stats.kurtosis(cD22)
            result.loc[idx, 'D3_KURT'] = stats.kurtosis(cD33)    

            result.loc[idx, 'A1_ZC'] = ZeroCrossing(cA11)
            result.loc[idx, 'A2_ZC'] = ZeroCrossing(cA22)
            result.loc[idx, 'A3_ZC'] = ZeroCrossing(cA33)
            result.loc[idx, 'D1_ZC'] = ZeroCrossing(cD11)
            result.loc[idx, 'D2_ZC'] = ZeroCrossing(cD22)
            result.loc[idx, 'D3_ZC'] = ZeroCrossing(cD33)   

            result.loc[idx, 'A1_SSC'] = SlopeSignChange(cA11)
            result.loc[idx, 'A2_SSC'] = SlopeSignChange(cA22)
            result.loc[idx, 'A3_SSC'] = SlopeSignChange(cA33)
            result.loc[idx, 'D1_SSC'] = SlopeSignChange(cD11)
            result.loc[idx, 'D2_SSC'] = SlopeSignChange(cD22)
            result.loc[idx, 'D3_SSC'] = SlopeSignChange(cD33)      

            cA11=extract_wave2(cA11)
            cD11=extract_wave2(cD11)
            cA22=extract_wave2(cA22)
            cD22=extract_wave2(cD22)
            cA33=extract_wave2(cA33)
            cD33=extract_wave2(cD33)

            result.loc[idx, 'A1_WL2'] = len(cA11)
            result.loc[idx, 'A2_WL2'] = len(cA22)
            result.loc[idx, 'A3_WL2'] = len(cA33)
            result.loc[idx, 'D1_WL2'] = len(cD11)
            result.loc[idx, 'D2_WL2'] = len(cD22)
            result.loc[idx, 'D3_WL2'] = len(cD33)

            result.loc[idx, 'A1_skew2'] = stats.skew(cA11)
            result.loc[idx, 'A2_skew2'] = stats.skew(cA22)
            result.loc[idx, 'A3_skew2'] = stats.skew(cA33)
            result.loc[idx, 'D1_skew2'] = stats.skew(cD11)
            result.loc[idx, 'D2_skew2'] = stats.skew(cD22)
            result.loc[idx, 'D3_skew2'] = stats.skew(cD33)

            result.loc[idx, 'A1_KURT2'] = stats.kurtosis(cA11)
            result.loc[idx, 'A2_KURT2'] = stats.kurtosis(cA22)
            result.loc[idx, 'A3_KURT2'] = stats.kurtosis(cA33)
            result.loc[idx, 'D1_KURT2'] = stats.kurtosis(cD11)
            result.loc[idx, 'D2_KURT2'] = stats.kurtosis(cD22)
            result.loc[idx, 'D3_KURT2'] = stats.kurtosis(cD33)       

            result.loc[idx, 'A1_ZC2'] = ZeroCrossing(cA11)
            result.loc[idx, 'A2_ZC2'] = ZeroCrossing(cA22)
            result.loc[idx, 'A3_ZC2'] = ZeroCrossing(cA33)
            result.loc[idx, 'D1_ZC2'] = ZeroCrossing(cD11)
            result.loc[idx, 'D2_ZC2'] = ZeroCrossing(cD22)
            result.loc[idx, 'D3_ZC2'] = ZeroCrossing(cD33) 

            result.loc[idx, 'A1_SSC2'] = SlopeSignChange(cA11)
            result.loc[idx, 'A2_SSC2'] = SlopeSignChange(cA22)
            result.loc[idx, 'A3_SSC2'] = SlopeSignChange(cA33)
            result.loc[idx, 'D1_SSC2'] = SlopeSignChange(cD11)
            result.loc[idx, 'D2_SSC2'] = SlopeSignChange(cD22)
            result.loc[idx, 'D3_SSC2'] = SlopeSignChange(cD33)  

            result.loc[idx, 'A1_CF2'] = SlopeSignChange(cA11)/(2*len(cA11))
            result.loc[idx, 'A2_CF2'] = SlopeSignChange(cA22)/(2*len(cA22))
            result.loc[idx, 'A3_CF2'] = SlopeSignChange(cA33)/(2*len(cA33))
            result.loc[idx, 'D1_CF2'] = SlopeSignChange(cD11)/(2*len(cD11))
            result.loc[idx, 'D2_CF2'] = SlopeSignChange(cD22)/(2*len(cD22))
            result.loc[idx, 'D3_CF2'] = SlopeSignChange(cD33)/(2*len(cD33))
            
        result = result.drop('index',axis=1) 
        return result