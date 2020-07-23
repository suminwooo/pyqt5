# PATH = 'sample_data/'
# file_name = 'SMND_345kV_EBG_A_S_0A_62_20190917184200_PRPS.csv'

####################################################################
import pandas as pd
import numpy as np
import pickle
import os
import csv
import time
import warnings
import sklearn
import sys
warnings.filterwarnings('ignore')

class multi_array_class:
    def Correl(array1, array2):
        m1 = np.mean(array1)
        m2 = np.mean(array2)
        CC = sum((array1 - m1) * (array2 - m2)) / np.sqrt(sum((array1 - m1) ** 2) * sum((array2 - m2) ** 2))
        return CC

class array_class:
    def Std(array):
        m = np.mean(array)
        n = len(array)
        std = np.sqrt(sum((array - m) ** 2) / (n - 1))
        return std
    def Skewness(array):
        m = np.mean(array)
        n = len(array)
        std = np.sqrt(sum((array - m) ** 2) / (n - 1))
        S = (n / ((n - 1) * (n - 2))) * sum(((array - m) / std) ** 3)
        if np.isnan(S) == True:
            S = 0
        return S
    def Kurtosis(array):
        m = np.mean(array)
        n = len(array)
        std = np.sqrt(sum((array - m) ** 2) / (n - 1))
        K = (((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * (sum(((array - m) / std) ** 4))) - 3 * ((n - 1) ** 2) / (
                    (n - 2) * (n - 3))
        return K
    def prps_to_prpd(array):
        prps_arr_t = np.transpose(array)

        amp_mapping_max = 256
        prpd_arr = np.array([[0] * amp_mapping_max for _ in range(256)])

        for c in range(3600):  # 3600(주기수) = c
            for m in range(256):  # 256(위상) = m
                amp_0_255_mapping_data = prps_arr_t[m][c]  # 256(진폭) = a

                if amp_0_255_mapping_data >= 255:
                    continue

                else:
                    prpd_arr[m][amp_0_255_mapping_data] += 1

        return prpd_arr

class num_class:
    def isNan(num):
        return (num != num).sum()

class PRPD_class:
    def unusual(PRPD):
        # column에 특이 케이스 (열이 모두 같은 경우가 있음.)
        if PRPD[:, 0].mean() != 0:
            if (PRPD[:, 0] == PRPD[:, 0].mean()).sum() == 256:
                PRPD = np.delete(PRPD, 0, axis=1)
        else:
            pass

        return PRPD
    def PRPD_preprocess(PRPD):
        zero_cnt = (PRPD == 0).sum(axis=0)

        for col in range(256):
            if zero_cnt[col] == 0:
                if zero_cnt[col + 1] - zero_cnt[col] > 0:
                    break

        cut_num = col + 2

        PRPD_pre = np.delete(PRPD, range(cut_num), axis=1)

        return PRPD_pre, cut_num

class mulit_PRPS_class :
    def PRPS_preprocess(PRPS, cut_num):
        PRPS_pre = np.where(PRPS <= cut_num, 0, PRPS)
        return PRPS_pre

class feature_class:
    def statics(feature):
        feature1 = feature[:128]
        feature2 = feature[128:]
        feature_CC = multi_array_class.Correl(feature1, feature2)
        feature1_K = array_class.Kurtosis(feature1)
        feature1_S = array_class.Skewness(feature1)
        feature2_K = array_class.Kurtosis(feature2)
        feature2_S = array_class.Skewness(feature2)
        feature1_std = np.std(feature1)
        feature2_std = np.std(feature2)
        return feature1_K, feature2_K, feature1_S, feature2_S, feature1_std, feature2_std, feature_CC

class opendata_class :
    def OpenData(path, file):
        element_list = []
        f = open(path + file, 'r', encoding='utf-8')
        rdr = csv.reader((x.replace('\0', '') for x in f))
        for line in rdr:
            if len(line) > 200:
                element_list.append(line[:-1])
        f.close()
        element_list = np.array(element_list).astype(int)
        return element_list
    def OpenData2(path, file):
        element_list = []
        f = open(path + file, 'r', encoding='utf-8')
        rdr = csv.reader((x.replace('\0', '') for x in f))
        for line in rdr:
            if len(line) > 200:
                element_list.append(line)
        f.close()
        element_list = np.array(element_list).astype(int)
        return element_list

# 모델 시작
def Model(PATH, file_name):
    try:
        PRPS = opendata_class.OpenData(PATH, file_name)
        PRPD = array_class.prps_to_prpd(PRPS)

    except:
        PRPS = opendata_class.OpenData2(PATH, file_name)
        PRPD = array_class.prps_to_prpd(PRPS)

    # column 특수케이스 처리
    PRPD = PRPD_class.unusual(PRPD)

    # PRPD 전처리
    PRPD_pre, cut_num = PRPD_class.PRPD_preprocess(PRPD)

    ################ PRPS ###################
    # PRPS 전처리
    PRPS_pre = mulit_PRPS_class.PRPS_preprocess(PRPS, cut_num)

    ############### Feature #################
    # feature A 통계값
    A = PRPD_pre.sum(axis=1)

    A1_K, A2_K, A1_S, A2_S, A1_std, A2_std, A_CC = feature_class.statics(A)

    # feature B 통계값
    B = np.array([])
    for row in PRPD_pre:
        B = np.append(B, array_class.Skewness(row))

    B1_K, B2_K, B1_S, B2_S, B1_std, B2_std, B_CC = feature_class.statics(B)

    # feature C 통계값
    C = np.array([])
    for col in PRPS_pre.T:
        C = np.append(C, np.max(col))
    Cn = C / np.max(C)

    Cn1_K, Cn2_K, Cn1_S, Cn2_S, Cn1_std, Cn2_std, Cn_CC = feature_class.statics(Cn)

    # feature D 통계값
    D = np.array([])
    for col in PRPS_pre.T:
        D = np.append(D, np.mean(col))
    Dm = D / np.max(C)

    Dm1_K, Dm2_K, Dm1_S, Dm2_S, Dm1_std, Dm2_std, Dm_CC = feature_class.statics(Dm)

    # Result
    result = [A1_K, A2_K, A1_S, A2_S, A1_std, A2_std, A_CC,
              B1_K, B2_K, B1_S, B2_S, B1_std, B2_std, B_CC,
              Cn1_K, Cn2_K, Cn1_S, Cn2_S, Cn1_std, Cn2_std, Cn_CC,
              Dm1_K, Dm2_K, Dm1_S, Dm2_S, Dm1_std, Dm2_std, Dm_CC]

    ############### Nan값이 들어있는 경우 다시 전처리 #################
    for i in result:
        if (i == i) == False:

            ###### cut num 5로 정의 #####
            re_cut_num = 5

            ###### PRPD 다시 전처리 #####
            PRPD_pre = np.delete(PRPD, range(re_cut_num), axis=1)

            ###### PRPS 다시 전처리 #####
            PRPS_pre = mulit_PRPS_class.PRPS_preprocess(PRPS, re_cut_num)

            ##### Feature값 다시 구하기 #####
            # feature A 통계값
            A = PRPD_pre.sum(axis=1)

            A1_K, A2_K, A1_S, A2_S, A1_std, A2_std, A_CC = feature_class.statics(A)

            # feature B 통계값
            B = np.array([])
            for row in PRPD_pre:
                B = np.append(B, array_class.Skewness(row))

            B1_K, B2_K, B1_S, B2_S, B1_std, B2_std, B_CC = feature_class.statics(B)

            # feature C 통계값
            C = np.array([])
            for col in PRPS_pre.T:
                C = np.append(C, np.max(col))
            Cn = C / np.max(C)

            Cn1_K, Cn2_K, Cn1_S, Cn2_S, Cn1_std, Cn2_std, Cn_CC = feature_class.statics(Cn)

            # feature D 통계값
            D = np.array([])
            for col in PRPS_pre.T:
                D = np.append(D, np.mean(col))
            Dm = D / np.max(C)

            Dm1_K, Dm2_K, Dm1_S, Dm2_S, Dm1_std, Dm2_std, Dm_CC = feature_class.statics(Dm)

            # Result
            result = [A1_K, A2_K, A1_S, A2_S, A1_std, A2_std, A_CC,
                      B1_K, B2_K, B1_S, B2_S, B1_std, B2_std, B_CC,
                      Cn1_K, Cn2_K, Cn1_S, Cn2_S, Cn1_std, Cn2_std, Cn_CC,
                      Dm1_K, Dm2_K, Dm1_S, Dm2_S, Dm1_std, Dm2_std, Dm_CC]

            result = pd.Series(result)
            result.loc[result.isna()] = 0
            result = list(result)

            break

        else:
            pass

    ################ 스케일링 ################
    scalerfile = 'data/scaler_data/using_scaler.sav'
    scaler = pickle.load(open(scalerfile, 'rb'))
    result = scaler.transform([np.array(result), np.array(result)])[0]

    ################ Model ################
    modelfile = 'data/model_data/using_model.sav'
    model = pickle.load(open(modelfile, 'rb'))
    Result = model.predict_proba(np.array([result, result]))[0]

    ######################## 분석 결과 저장 ##########################
    now = time.localtime()

    head, tail = os.path.splitext(file_name)
    FileName = head
    CREAT_TIME = "%04d/%02d/%02d %02d:%02d:%02d" % (
    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    svm_df = pd.DataFrame({'FileName': FileName,
                           'CREAT_TIME': CREAT_TIME,
                           'VOID_JDGM': (Result[3] * 100).round(2),
                           'SURFACE_JDGM': (Result[2] * 100).round(2),
                           'CORONA_JDGM': (Result[0] * 100).round(2),
                           'NOISE_JDGM': (Result[1] * 100).round(2)}, index=[0])

    return [(Result[3] * 100).round(2), (Result[2] * 100).round(2), (Result[0] * 100).round(2), (Result[1] * 100).round(2)]

