import pandas as pd
import numpy as np
import pickle
import time
import os
import math
import glob
from scipy.stats import kurtosis
from scipy.stats import skew
import csv
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


def score_model(model, params, cv=None):
    if cv is None:
        cv = StratifiedKFold(n_splits=5, random_state=1004, shuffle=True)

    smoter = SMOTE(random_state=42)

    scores = []

    for train_fold_index, val_fold_index in cv.split(XX, yy):
        X1, y1 = XX.loc[train_fold_index], yy.loc[train_fold_index]
        X2, y2 = X_shift4.loc[train_fold_index], y_shift4.loc[train_fold_index]
        X3, y3 = X_shift8.loc[train_fold_index], y_shift8.loc[train_fold_index]
        X4, y4 = X_shift12.loc[train_fold_index], y_shift12.loc[train_fold_index]

        X_train_fold = pd.concat([X1, X2, X3, X4]).reset_index(drop=True)
        X_train_fold = pd.DataFrame(scaler.transform(X_train_fold))
        y_train_fold = pd.concat([y1, y2, y3, y4]).reset_index(drop=True)

        X_val_fold, y_val_fold = XX.loc[val_fold_index], yy.loc[val_fold_index]

        X_val_fold = pd.DataFrame(scaler.transform(X_val_fold))

        X_train_fold_upsample, y_train_fold_upsample = smoter.fit_resample(X_train_fold,
                                                                           y_train_fold)

        model_obj = model(**params).fit(X_train_fold_upsample, y_train_fold_upsample)
        score = accuracy_score(y_val_fold, model_obj.predict(X_val_fold))
        scores.append(score)
    return np.array(scores)


def Correl(array1, array2):
    m1 = np.mean(array1)
    m2 = np.mean(array2)
    CC = sum((array1 - m1) * (array2 - m2)) / np.sqrt(sum((array1 - m1) ** 2) * sum((array2 - m2) ** 2))
    return CC


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


def isNan(num):
    return (num != num).sum()


def unusual(PRPD):
    # column에 특이 케이스 (첫줄이 모두 같은 경우가 있음.)
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


def PRPS_preprocess(PRPS, cut_num):
    PRPS_pre = np.where(PRPS <= cut_num, 0, PRPS)

    return PRPS_pre


def statics(feature):
    feature1 = feature[:128]
    feature2 = feature[128:]

    feature_CC = Correl(feature1, feature2)
    feature1_K = Kurtosis(feature1)
    feature1_S = Skewness(feature1)
    feature2_K = Kurtosis(feature2)
    feature2_S = Skewness(feature2)
    feature1_std = np.std(feature1)
    feature2_std = np.std(feature2)

    return feature1_K, feature2_K, feature1_S, feature2_S, feature1_std, feature2_std, feature_CC


def OpenData(path_file):
    element_list=[]
    f = open(path_file, 'r', encoding='utf-8')
    rdr = csv.reader((x.replace('\0', '') for x in f))
    for line in rdr:
        if len(line)>200:
            element_list.append(line)
    f.close()
    element_list = np.array(element_list).astype(int)
    return element_list

def OpenData2(path_file):
    element_list=[]
    f = open(path_file, 'r', encoding='utf-8')
    rdr = csv.reader((x.replace('\0', '') for x in f))
    for line in rdr:
        if len(line)>200:
            element_list.append(line[:-1])
    f.close()
    element_list = np.array(element_list).astype(int)
    return element_list


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

print('시작...')
folders = os.listdir('data/update_data')

dic = {}

for folder in folders:
    if 'void' in folder.lower():
        dic[folder] = 'void'
    elif 'surface' in folder.lower():
        dic[folder] = 'surface'
    elif 'corona' in folder.lower():
        dic[folder] = 'corona'
    else:
        dic[folder] = 'noise'

Total = []

for folder in folders:
    PATH = 'data/update_data/{}/'.format(folder)

    for idx in range(len(glob.glob(PATH + '*PRPS*'))):

        # PRPS 불러오기
        PRPS = OpenData(glob.glob(PATH + '*PRPS*')[idx])

        if PRPS.shape != (3600, 256):
            PRPS = OpenData2(glob.glob(PATH + '*PRPS*')[idx])
            print('error-1')

        else:
            pass

        # PRPS 위상변환
        PRPS0 = PRPS.copy()
        PRPS4 = np.hstack([PRPS[:, 4:], PRPS[:, :4]])
        PRPS8 = np.hstack([PRPS[:, 8:], PRPS[:, :8]])
        PRPS12 = np.hstack([PRPS[:, 12:], PRPS[:, :12]])

        for num, prps in enumerate([PRPS0, PRPS4, PRPS8, PRPS12]):

            # PRPD 변환
            PRPD = prps_to_prpd(prps)

            # column 특수케이스 전처리
            PRPD = unusual(PRPD)

            # PRPD 전처리
            PRPD_pre, cut_num = PRPD_preprocess(PRPD)

            # PRPS 전처리
            PRPS_pre = PRPS_preprocess(prps, cut_num)

            # feature A 통계값
            A = PRPD_pre.sum(axis=1)

            A1_K, A2_K, A1_S, A2_S, A1_std, A2_std, A_CC = statics(A)

            # feature B 통계값
            B = np.array([])
            for row in PRPD_pre:
                B = np.append(B, Skewness(row))

            B1_K, B2_K, B1_S, B2_S, B1_std, B2_std, B_CC = statics(B)

            # feature C 통계값
            C = np.array([])
            for col in PRPS_pre.T:
                C = np.append(C, np.max(col))
            Cn = C / np.max(C)

            Cn1_K, Cn2_K, Cn1_S, Cn2_S, Cn1_std, Cn2_std, Cn_CC = statics(Cn)

            # feature D 통계값
            D = np.array([])
            for col in PRPS_pre.T:
                D = np.append(D, np.mean(col))
            Dm = D / np.max(C)

            Dm1_K, Dm2_K, Dm1_S, Dm2_S, Dm1_std, Dm2_std, Dm_CC = statics(Dm)

            # 방전유형
            discharge = dic[folder]

            # Result
            result = [str(folder) + '_' + str(glob.glob(PATH + '*PRPS*')[idx].split('_')[-2]),
                      num, discharge,
                      A1_K, A2_K, A1_S, A2_S, A1_std, A2_std, A_CC,
                      B1_K, B2_K, B1_S, B2_S, B1_std, B2_std, B_CC,
                      Cn1_K, Cn2_K, Cn1_S, Cn2_S, Cn1_std, Cn2_std, Cn_CC,
                      Dm1_K, Dm2_K, Dm1_S, Dm2_S, Dm1_std, Dm2_std, Dm_CC]

            for i in result:
                if (i == i) == False:
                    print('error-2')

                    # cut num 5로 정의
                    re_cut_num = 5

                    # PRPD 다시 전처리
                    PRPD_pre = np.delete(PRPD, range(re_cut_num), axis=1)

                    # PRPS 다시 전처리
                    PRPS_pre = PRPS_preprocess(prps, re_cut_num)

                    # feature A 통계값
                    A = PRPD_pre.sum(axis=1)

                    A1_K, A2_K, A1_S, A2_S, A1_std, A2_std, A_CC = statics(A)

                    # feature B 통계값
                    B = np.array([])
                    for row in PRPD_pre:
                        B = np.append(B, Skewness(row))

                    B1_K, B2_K, B1_S, B2_S, B1_std, B2_std, B_CC = statics(B)

                    # feature C 통계값
                    C = np.array([])
                    for col in PRPS_pre.T:
                        C = np.append(C, np.max(col))
                    Cn = C / np.max(C)

                    Cn1_K, Cn2_K, Cn1_S, Cn2_S, Cn1_std, Cn2_std, Cn_CC = statics(Cn)

                    # feature D 통계값
                    D = np.array([])
                    for col in PRPS_pre.T:
                        D = np.append(D, np.mean(col))
                    Dm = D / np.max(C)

                    Dm1_K, Dm2_K, Dm1_S, Dm2_S, Dm1_std, Dm2_std, Dm_CC = statics(Dm)

                    # 방전유형
                    discharge = dic[folder]

                    # Result
                    result = [str(folder) + '_' + str(glob.glob(PATH + '*PRPS*')[idx].split('_')[-2]),
                              num, discharge,
                              A1_K, A2_K, A1_S, A2_S, A1_std, A2_std, A_CC,
                              B1_K, B2_K, B1_S, B2_S, B1_std, B2_std, B_CC,
                              Cn1_K, Cn2_K, Cn1_S, Cn2_S, Cn1_std, Cn2_std, Cn_CC,
                              Dm1_K, Dm2_K, Dm1_S, Dm2_S, Dm1_std, Dm2_std, Dm_CC]

                    result = pd.Series(result)
                    result.loc[result.isna()] = 0
                    result = list(result)

                    break

                else:
                    pass

                    # 데이터 저장
            Total.append(result)

df = pd.DataFrame(Total,
                  columns=['location', 'shift', 'discharge',
                           'A1_K', 'A2_K', 'A1_S', 'A2_S', 'A1_std', 'A2_std', 'A_CC',
                           'B1_K', 'B2_K', 'B1_S', 'B2_S', 'B1_std', 'B2_std', 'B_CC',
                           'Cn1_K', 'Cn2_K', 'Cn1_S', 'Cn2_S', 'Cn1_std', 'Cn2_std', 'Cn_CC',
                           'Dm1_K', 'Dm2_K', 'Dm1_S', 'Dm2_S', 'Dm1_std', 'Dm2_std',
                           'Dm_CC'])  # .to_csv('Total2.csv',index=False)

shift0 = df[df['shift'] == 0].iloc[:, 2:].reset_index(drop=True)
shift4 = df[df['shift'] == 1].iloc[:, 2:].reset_index(drop=True)
shift8 = df[df['shift'] == 2].iloc[:, 2:].reset_index(drop=True)
shift12 = df[df['shift'] == 3].iloc[:, 2:].reset_index(drop=True)

sampley = shift0['discharge']
sampleX = shift0.drop('discharge', axis=1)

sc = pd.concat([shift0.drop('discharge', axis=1), shift4.drop('discharge', axis=1), shift8.drop('discharge', axis=1),
                shift12.drop('discharge', axis=1)]).reset_index(drop=True)
print('sclaer 만들기 시작...')

scaler = StandardScaler()
scaler.fit(sc)

scalerfile = 'new_scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))

XX, X_test, yy, y_test = train_test_split(sampleX, sampley, test_size=0.2, random_state=1004, stratify=sampley)

XX_idx = XX.index
yy = yy.reset_index(drop=True)
XX = XX.reset_index(drop=True)

shift4 = shift4.loc[XX_idx].reset_index(drop=True)
shift8 = shift8.loc[XX_idx].reset_index(drop=True)
shift12 = shift12.loc[XX_idx].reset_index(drop=True)

y_shift4 = shift4['discharge']
X_shift4 = shift4.drop('discharge', axis=1)
y_shift8 = shift8['discharge']
X_shift8 = shift8.drop('discharge', axis=1)
y_shift12 = shift12['discharge']
X_shift12 = shift12.drop('discharge', axis=1)

param_grid = {'gamma': [100, 1000],
              'C': [1, 10],
              'kernel': ['rbf']}

score_tracker = []

for gamma in param_grid['gamma']:
    for C in param_grid['C']:
        for kernel in param_grid['kernel']:
            example_params = {
                'gamma': gamma,
                'C': C,
                'kernel': kernel,
                'random_state': 1004
            }
            example_params['accuracy'] = score_model(SVC,
                                                     example_params).mean()

            score_tracker.append(example_params)

ls = []
for i in range(len(score_tracker)):
    ls.append(score_tracker[i]['accuracy'])
gamma = score_tracker[ls.index(max(ls))]['gamma']
C = score_tracker[ls.index(max(ls))]['C']

svm = SVC(C=C, gamma=gamma, random_state=1004)

XX = scaler.transform(XX)
X_test = scaler.transform(X_test)

smoter = SMOTE(random_state=42)
XX_train, yy_train = smoter.fit_resample(XX, yy)

clf = CalibratedClassifierCV(svm)
clf.fit(XX_train, yy_train)

print('모델 만들기 시작...')
modelfile = 'new_model.sav'
pickle.dump(clf, open(modelfile, 'wb'))