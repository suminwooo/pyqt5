# -*- encoding : utf-8 -*-

import sys

sys.setrecursionlimit(5000)
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import time
from DataAnalysis_ import track1
from Track3_Analysis import ANALYZER, track23_popup
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets
import xlrd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

track2_result = 0


# 에러로 인한 강제종료 막는 함수
def my_exception_hook(exctype, value, traceback):
    sys._excepthook(exctype, value, traceback)


sys._excepthook = sys.excepthook
sys.excepthook = my_exception_hook

# 전역 변수 설정
pca_result = 0
topology_max = 0


class popup_screen(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def model_result(self, original_box):
        group_box_void = QGroupBox('void')
        group_box_surface = QGroupBox('surface')
        group_box_corona = QGroupBox('corona')
        group_box_noise = QGroupBox('noise')
        self.void_text = QLabel('num')
        self.surface_text = QLabel('num')
        self.corona_text = QLabel('num')
        self.noise_text = QLabel('num')
        void_detail_layout = QGridLayout()
        void_detail_layout.addWidget(self.void_text)
        group_box_void.setLayout(void_detail_layout)
        void_layout = QVBoxLayout()
        void_layout.addWidget(group_box_void)
        surface_detail_surface = QGridLayout()
        surface_detail_surface.addWidget(self.surface_text)
        group_box_surface.setLayout(surface_detail_surface)
        surface_layout = QVBoxLayout()
        surface_layout.addWidget(group_box_surface)
        corona_deail_layout = QGridLayout()
        corona_deail_layout.addWidget(self.corona_text)
        group_box_corona.setLayout(corona_deail_layout)
        corona_layout = QVBoxLayout()
        corona_layout.addWidget(group_box_corona)
        noise_detail_layout = QGridLayout()
        noise_detail_layout.addWidget(self.noise_text)
        group_box_noise.setLayout(noise_detail_layout)
        noise_layout = QVBoxLayout()
        noise_layout.addWidget(group_box_noise)
        detail_layout = QGridLayout()
        detail_layout.addWidget(group_box_void, 0, 0)
        detail_layout.addWidget(group_box_surface, 0, 1)
        detail_layout.addWidget(group_box_corona, 1, 0)
        detail_layout.addWidget(group_box_noise, 1, 1)
        original_box.setLayout(detail_layout)
        layout = QVBoxLayout()
        layout.addWidget(original_box)

    def button_block(self):
        self.range_box_label.setEnabled(False)

    def range_clicked(self):  # 버튼 눌렀을때 나타나는 이벤트
        setting_value = (MyWindow.__dict__['_MyWindow__shared_state']['setting_clustering_list'])

        self.value_list = [setting_value]
        return self.range_labeling.setText('입력한 클러스터링 번호 : {}'.format(self.value_list[0]))

    def calculate(self):
        pca_result = MyWindow.__dict__['_MyWindow__shared_state']['pca_result']
        topology_max = MyWindow.__dict__['_MyWindow__shared_state']['topology_max']
        labels = MyWindow.__dict__['_MyWindow__shared_state']['labels']
        clustering_number = MyWindow.__dict__['_MyWindow__shared_state']['setting_clustering_list']
        clustering_number = [int(i) for i in clustering_number.split(',')]

        filter_value = track23_popup().Filter(pca_result=pca_result, topology_max=topology_max, labels=labels,
                                              choise_cluster_list=clustering_number)
        self.filter_value_result = filter_value[0]
        self.filter_value_filter_data = filter_value[1]

        self.fig.clear()

        ####
        # import numpy as np
        # from scipy.stats import gaussian_kde
        # x = self.filter_value_filter_data['Max']
        # y = self.filter_value_filter_data['topology']
        # # Calculate the point density
        # xy = np.vstack([x, y])
        # z = gaussian_kde(xy)(xy)
        # # Sort the points by density, so that the densest points are plotted last
        # idx = z.argsort()
        # x, y, z = x[idx], y[idx], z[idx]
        # for i in range(len(x)):
        #     ax.scatter(x[i], y[i], s=50)
        ax = self.fig.add_subplot()
        ax.hist2d(self.filter_value_filter_data['Max'], self.filter_value_filter_data['topology'], (50, 50),
                  cmap='bone_r')
        self.canvas.draw()
        svm_df = track23_popup().Classifier(self.filter_value_result)

        self.void_text.setText('{} %'.format(float(svm_df['VOID_JDGM'])))
        self.surface_text.setText('{} %'.format(float(svm_df['SURFACE_JDGM'])))
        self.corona_text.setText('{} %'.format(float(svm_df['CORONA_JDGM'])))
        self.noise_text.setText('{} %'.format(float(svm_df['NOISE_JDGM'])))

    def setupUI(self):
        # 박스 설정
        self.setWindowTitle("graph detail")
        self.full_group_box = QGroupBox('전체 ')
        self.range_group_box = QGroupBox('범위')
        self.range_box_label = QPushButton('버튼을 눌러서 클러스터링 번호를 확인하세요')
        self.range_labeling = QLabel('버튼을 눌러서 클러스터링 번호를 확인하세요')
        self.range_box_label.clicked.connect(self.range_clicked)
        self.range_box_label.clicked.connect(self.button_block)

        self.graph_box = QGroupBox("그래프")
        self.graph_btn = QPushButton("그래프 생성 버튼")
        self.graph_btn.clicked.connect(self.calculate)
        self.fig = plt.Figure()  # 그림부분
        self.canvas = FigureCanvas(self.fig)
        self.result_box = QGroupBox("패턴 인식 결과")
        self.model_result(self.result_box)  # 값 넣기

        # 레이아웃 설정

        self.layout_all = QVBoxLayout()  # 전체 레이아웃
        self.layout_detail = QVBoxLayout()  # 전체 디테일 레이아웃
        self.range_layout = QVBoxLayout()  # 범위설정 부분
        self.range_layout.addWidget(self.range_box_label)
        self.range_layout.addWidget(self.range_labeling)
        self.range_group_box.setLayout(self.range_layout)
        self.grahp_layout = QVBoxLayout()  # 그래프 부분
        self.layout_graph_btn = QVBoxLayout()  # 오른쪽 두번째 그래프 부분 레이아웃
        self.layout_graph_btn.addWidget(self.graph_btn, alignment=QtCore.Qt.AlignLeft)
        self.layout_graph_btn.addWidget(self.canvas)
        self.graph_box.setLayout(self.layout_graph_btn)
        self.graph_box.setLayout(self.grahp_layout)
        self.grahp_layout.addWidget(self.graph_box)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.range_group_box, 1)
        self.layout.addWidget(self.graph_box, 10)
        self.layout.addWidget(self.result_box, 1)
        self.setLayout(self.layout)

        # 끝
        self.setGeometry(200, 200, 750, 850)


class popup_screen_dendrogram(QDialog):
    def __init__(self, parent=None):
        # super().__init__(parent)
        pca_result = None
        self.dendrogram()

    def dendrogram(self):
        pca_result = (MyWindow.__dict__['_MyWindow__shared_state']['pca_result'])[0]
        mergings = linkage(pca_result, method='ward')
        dendrogram(mergings)
        plt.xticks(color='w')
        plt.show()


class file_path(QListWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentItemChanged.connect(self.select_file_name)
        self.setAcceptDrops(True)
        self.links_list = []
        self.file_name = []

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            links_list = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    links_list.append(str(url.toLocalFile()).split('/')[-1])
                else:
                    links_list.append(str(url.toLocalFile()).split('/')[-1])
            self.addItems(links_list)
        else:
            event.ignore()

    def select_file_name(self, item):
        file_name = '{}'.format(item.text())
        self.file_name.append(file_name)
        return self.file_name


class MyWindow(QWidget):
    __shared_state = {"setting_clustering_list": '범위X', "pca_result": 'pca_result', "topology_max": 'topology_max',
                      "labels": 'labels', "clustering_number": 0}

    # setting_clustering_list : track2 마지막 부분에 직접 입력한 리스트의 수
    # clustering_number : track2 첫 부분에 클러스터링 하는 수
    def __init__(self):
        self.__dict__ = self.__shared_state
        pass
        super().__init__()
        self.file_path1 = file_path()
        self.file_path2 = file_path()
        self.file_path_all = file_path()
        self.setupUI()

    def closeEvent(self, event):  # 프로그램 닫을때 나타나는 이벤트
        message = QMessageBox.question(self, "Message", "종료??", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if message == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def edit_result(self):
        # import sklearn
        self.track1_model = track1('data/sample_data/', self.file_name1)
        self.value = self.track1_model.Model()[0]
        self.void_text_track1.setText('{} %'.format(float(self.value[0])))
        self.surface_text_track1.setText('{} %'.format(float(self.value[1])))
        self.corona_text_track1.setText('{} %'.format(float(self.value[2])))
        self.noise_text_track1.setText('{} %'.format(float(self.value[3])))
        return self.value[-1]

    def popup_btn(self):
        mydialog = popup_screen(self)
        mydialog.show()

    def popup_dendrogram_btn(self):
        popup_screen_dendrogram(self)

    def check_clicked(self):  # 버튼 눌렀을때 나타나는 이벤트
        QMessageBox.about(self, "message", 'ㅇ눌러짐')

    def check_clicked_range(self):  # 버튼 눌렀을때 나타나는 이벤트
        QMessageBox.about(self, "message", 'ㅇ범위셋팅됨')

    def track1_file_clicked(self):  # 선택한 파일 확인
        QMessageBox.about(self, "message", self.file_path1.file_name[-1] + ' 선택됨')
        self.file_name1 = self.file_path1.file_name[-1]

    def update_model_scaler_btn(self):
        QMessageBox.about(self, "message", '모델에 따라 시간이 오래 걸릴 수 있습니다.')
        import DataUpdate_

    def track2_file_clicked(self):  # 선택한 파일 확인
        self.file_name2 = self.file_path2.file_name[-1]
        return self.file_name2

    def track2_file_modeling(self):
        data_analysis_class = ANALYZER('data/sample_data', self.track2_file_clicked())
        QMessageBox.about(self, "message", self.file_path2.file_name[-1] + '실행')
        data_analysis = data_analysis_class.Extractor(data_analysis_class.LoadData())
        self.pca_result = [data_analysis[0]]
        self.topology_max = [data_analysis[1]]
        self.__shared_state['pca_result'] = self.pca_result
        self.__shared_state['topology_max'] = self.topology_max

    def show_graph_cluster_list_show(self):
        data_analysis_class = ANALYZER('data/sample_data', self.track2_file_clicked())
        dendrogram_data = data_analysis_class.Cluster((MyWindow.__dict__['_MyWindow__shared_state']['pca_result'])[0],
                                                      self.clustering_number_btn())
        self.labels = dendrogram_data[0]
        self.__shared_state['labels'] = self.labels
        self.labels = self.__shared_state['labels']
        self.cluster_list = list(dendrogram_data[1])
        self.tsne = dendrogram_data[2]
        try:
            self.value_box.clear()
            self.graph_cluster()
            self.list_cluster()
        except:
            self.graph_cluster()
            self.list_cluster()

    def list_cluster(self):
        # 리스트 뿌려주기
        self.cluster_list_box_layout = QVBoxLayout()  # 클러스터 리스트 레이아웃
        for num, i in enumerate(range(len(self.cluster_list))):
            self.value_box = QLabel()
            self.value_box.setText('{} cluster : {}'.format(num, self.cluster_list[i]))
            self.cluster_list_box_layout.addWidget(self.value_box)
            self.clustering_list_track23.setLayout(self.cluster_list_box_layout)
            # self.layout_second_graph_btn_track23.addWidget(self.clustering_list_track23, 2)

    def graph_cluster(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        for color in range(len(set(self.tsne['color']))):
            x = self.tsne[self.tsne['color'] == color]['x']
            y = self.tsne[self.tsne['color'] == color]['y']
            ax.scatter(x, y, alpha=0.3, label='{}'.format(color))
        ax.legend()
        self.canvas.draw()

    def clustering_number_btn(self):
        self.fig.clear()
        self.number = self.lineedit_clustering_count_track23.text()
        self.number = int(self.number)
        self.__shared_state['clustering_number'] = self.number
        number = self.__shared_state['clustering_number']

        return number

    def save_range_value(self):
        self.setting_clustering_list = self.x_max_line.text()

    def preprocessing_result_track1(self):
        # track1_model = track1('data/sample_data/', self.file_name1)
        each_value = self.track1_model.Model()[-1]
        now = time.localtime()
        CREAT_TIME = "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday,
                                                        now.tm_hour, now.tm_min, now.tm_sec)
        data_dic = {'FileName': self.file_name1, 'CREAT_TIME': CREAT_TIME}
        value_list = ['A1_K', 'A2_K', 'A1_S', 'A2_S', 'A1_std', 'A2_std', 'A_CC', 'B1_K', 'B2_K', 'B1_S', 'B2_S',
                      'B1_std', 'B2_std', 'B_CC', 'Cn1_K', 'Cn2_K', 'Cn1_S', 'Cn2_S', 'Cn1_std', 'Cn2_std', 'Cn_CC',
                      'Dm1_K', 'Dm2_K', 'Dm1_S', 'Dm2_S', 'Dm1_std', 'Dm2_std', 'Dm_CC']
        for i, j in zip(value_list, each_value):
            data_dic['{}'.format(i)] = j
        svm_df = pd.DataFrame(data_dic, index=[0])
        svm_df.to_csv('data/{}_Result.csv'.format(self.file_path1.file_name[-1][:-4]), index=False)

    def setupUI(self):
        ### 박스 설정
        # 왼쪽
        self.track1_group_box = QGroupBox('Track1')
        self.csv_group_box_track1 = QGroupBox()
        self.csv_group_box_drop_track1 = QGroupBox('특정 폴더에 csv 파일 떨구기기')
        self.delete_btn_track1 = QPushButton('파일 선택 버튼(필수! 파일 선택후 눌러주기)')
        self.delete_btn_track1.clicked.connect(self.track1_file_clicked)
        self.data_btn_track1 = QPushButton('머신러닝 test_data_input 버튼')
        self.data_btn_track1.clicked.connect(self.check_clicked)
        self.data_btn_track1.clicked.connect(self.edit_result)  # 값 넣기
        self.result_group_box_track1 = QGroupBox('패턴 인식 결과')
        self.result_label_void_track1 = QLabel('void')
        self.result_label_surface_track1 = QLabel('surface')
        self.result_label_corona_track1 = QLabel('corona')
        self.result_label_noise_track1 = QLabel('noise')
        self.void_text_track1 = QLabel('nan')
        self.surface_text_track1 = QLabel('nan')
        self.corona_text_track1 = QLabel('nan')
        self.noise_text_track1 = QLabel('nan')
        self.result_btn_track1 = QPushButton('전처리 결과 Excel 출력 버튼')
        self.result_btn_track1.clicked.connect(self.check_clicked)
        self.result_btn_track1.clicked.connect(self.preprocessing_result_track1)  # 값 넣기
        self.model_updata_btn_track1 = QPushButton('업데이트 버튼')
        self.model_updata_btn_track1.clicked.connect(self.update_model_scaler_btn)
        self.empy_groupbox = QGroupBox('made by sumin.w')
        self.empy_groupbox.setStyleSheet("border:0;")
        # 오른쪽 그룹박스
        self.track23_group_box = QGroupBox('Track2&&3')
        self.csv_group_box_track23 = QGroupBox()
        self.csv_group_box_drop_track23 = QGroupBox('특정 폴더에 csv 파일 떨구기기')
        self.csv_group_check_btn_track23 = QPushButton('◁◁◁파일 선택 버튼')
        self.csv_group_check_btn_track23.clicked.connect(self.track2_file_clicked)
        self.csv_group_check_btn_track23.clicked.connect(self.track2_file_modeling)
        self.dendrogram_btn_track23 = QPushButton('덴드로그램 팝업 버튼')
        self.dendrogram_btn_track23.clicked.connect(self.popup_dendrogram_btn)
        self.file_path1.selectedItems()
        self.axis_track23 = QGroupBox('클러스터링')
        self.clustering_count_label_track23 = QLabel('클러스터링 입력 수 ')
        self.lineedit_clustering_count_track23 = QLineEdit()
        self.reset_count_btn_track23 = QPushButton('클러스터링 수 입력')
        self.reset_count_btn_track23.clicked.connect(self.clustering_number_btn)
        self.center_graph_btn_track23 = QPushButton('클러스터링 리스트 및 그래프 출력 버튼')
        self.center_graph_btn_track23.clicked.connect(self.show_graph_cluster_list_show)
        self.graph_axis_track23 = QGroupBox()
        self.center_detail_graph_track23 = QGroupBox('각 축 좌표 범위 검색 및 이미지 출력')
        self.fig = plt.Figure()  # 그림부분
        self.canvas = FigureCanvas(self.fig)
        self.clustering_list_track23 = QGroupBox('클러스터 리스트')
        self.three_graph_track23 = QGroupBox('클러스터링 입력 부분  # 리스트 형태로 적어주세요. (EX. 1,2,3,4)')
        self.search_xyz_track23 = QGroupBox('좌표 검색')
        self.setting_value_track23 = QLineEdit('')
        self.axis_setting_btn_track23 = QPushButton('클러스터링 입력')
        self.axis_setting_btn_track23.clicked.connect(self.check_clicked_range)
        self.axis_setting_btn_track23.clicked.connect(self.save_range_value)
        self.x_max_line = self.setting_value_track23
        self.graph_import_btn_track23 = QPushButton('클러스터링 그래프 출력')
        self.graph_import_btn_track23.clicked.connect(self.popup_btn)
        #### 레이아웃
        self.main_layout = QHBoxLayout()  # 메인 레이아웃
        ## 왼쪽 레이아웃
        self.layout_track1 = QVBoxLayout()  # 왼쪽 레이아웃
        self.layout_detail_track1 = QVBoxLayout()  # 왼쪽 디테일 레이아웃
        self.layout_first_drop_track1 = QVBoxLayout()  # 왼쪽 첫번째 레이아웃
        self.layout_drop_file_track1 = QVBoxLayout()  # 왼쪽 파일 떨구는 레이아웃
        self.layout_drop_file_track1.addWidget(self.file_path1)
        self.layout_drop_file_track1.addWidget(self.delete_btn_track1)
        self.csv_group_box_drop_track1.setLayout(self.layout_drop_file_track1)
        self.layout_first_drop_track1.addWidget(self.csv_group_box_drop_track1)
        self.csv_group_box_track1.setLayout(self.layout_first_drop_track1)  # 인풋 버튼
        self.detail_layout_track1 = QGridLayout()
        self.detail_layout_track1.addWidget(self.result_label_void_track1, 0, 0)
        self.detail_layout_track1.addWidget(self.result_label_surface_track1, 1, 0)
        self.detail_layout_track1.addWidget(self.result_label_corona_track1, 2, 0)
        self.detail_layout_track1.addWidget(self.result_label_noise_track1, 3, 0)
        self.detail_layout_track1.addWidget(self.void_text_track1, 0, 1)
        self.detail_layout_track1.addWidget(self.surface_text_track1, 1, 1)
        self.detail_layout_track1.addWidget(self.corona_text_track1, 2, 1)
        self.detail_layout_track1.addWidget(self.noise_text_track1, 3, 1)
        self.result_group_box_track1.setLayout(self.detail_layout_track1)
        self.final_result_layout_track1 = QVBoxLayout()
        self.final_result_layout_track1.addWidget(self.result_group_box_track1)
        self.layout_drop_file_update_track1 = QVBoxLayout()  # 왼쪽 전체 업데이트 파일 떨구는 레이아웃
        self.layout_drop_file_update_track1.addWidget(self.file_path_all)
        self.layout_detail_track1.addWidget(self.csv_group_box_track1, 1)
        self.layout_detail_track1.addWidget(self.data_btn_track1, 1)
        self.layout_detail_track1.addWidget(self.result_group_box_track1, 1)
        self.layout_detail_track1.addWidget(self.result_btn_track1, 1)
        self.layout_detail_track1.addWidget(self.model_updata_btn_track1, 1)
        self.layout_detail_track1.addWidget(self.empy_groupbox, 1)
        self.track1_group_box.setLayout(self.layout_detail_track1)
        self.layout_track1.addWidget(self.track1_group_box)
        self.main_layout.addLayout(self.layout_track1, 1)
        ## 오른쪽 레이아웃
        self.layout_track23 = QVBoxLayout()  # 오른쪽 레이아웃
        self.layout_detail_track23 = QVBoxLayout()  # 오른쪽 디테일 레이아웃
        self.layout_second_track23 = QHBoxLayout()  # 오른쪽 첫번째 레이아웃
        self.layout_drop_file_track23 = QVBoxLayout()  # 오른쪽 파일 떨구는 레이아웃
        self.layout_drop_file_track23.addWidget(self.file_path2)
        self.csv_group_box_drop_track23.setLayout(self.layout_drop_file_track23)
        self.layout_axis_track23 = QGridLayout()  # 클러스터링 레이아웃
        self.layout_axis_track23.addWidget(self.clustering_count_label_track23, 0, 0)
        self.layout_axis_track23.addWidget(self.lineedit_clustering_count_track23, 0, 1)
        self.layout_axis_track23.addWidget(self.reset_count_btn_track23, 1, 0)
        self.layout_axis_track23.addWidget(self.center_graph_btn_track23, 1, 1)
        self.axis_track23.setLayout(self.layout_axis_track23)
        self.layout_second_track23.addWidget(self.csv_group_box_drop_track23, 2)
        self.layout_second_track23.addWidget(self.csv_group_check_btn_track23, 1)
        self.layout_second_track23.addWidget(self.dendrogram_btn_track23, 1)
        self.layout_second_track23.addWidget(self.axis_track23, 10)
        self.csv_group_box_track23.setLayout(self.layout_second_track23)
        self.layout_second_track23 = QHBoxLayout()  # 오른쪽 두번째 레이아웃
        self.layout_second_graph_btn_track23 = QHBoxLayout()  # 오른쪽 두번째 그래프 부분 레이아웃
        self.cluster_list_box_layout = QVBoxLayout()  # 클러스터 리스트 레이아웃
        self.clustering_list_track23.setLayout(self.cluster_list_box_layout)
        self.layout_second_graph_btn_track23.addWidget(self.canvas, 8)
        self.layout_second_graph_btn_track23.addWidget(self.clustering_list_track23, 2)
        self.graph_axis_track23.setLayout(self.layout_second_graph_btn_track23)
        self.graph_axis_track23.setLayout(self.layout_second_track23)
        self.layout_second_track23.addWidget(self.graph_axis_track23)
        self.layout_third_track23 = QHBoxLayout()  # 오른쪽 세번째 레이아웃
        self.layout_third_3_detail_1_track23 = QHBoxLayout()
        self.layout_third_3_detail_1_track23.addWidget(self.setting_value_track23)
        self.layout_third_track23.addLayout(self.layout_third_3_detail_1_track23)
        self.layout_third_track23.addWidget(self.axis_setting_btn_track23)
        self.layout_third_track23.addWidget(self.graph_import_btn_track23)
        self.three_graph_track23.setLayout(self.layout_third_track23)
        self.layout_third_track23.addWidget(self.three_graph_track23)
        self.layout_detail_track23.addWidget(self.csv_group_box_track23, 1)
        self.layout_detail_track23.addWidget(self.graph_axis_track23, 10)
        self.layout_detail_track23.addWidget(self.three_graph_track23, 1)
        self.track23_group_box.setLayout(self.layout_detail_track23)
        self.layout_track23.addWidget(self.track23_group_box)
        self.main_layout.addLayout(self.layout_track23, 6)
        ##### 마무리
        self.setLayout(self.main_layout)
        self.setWindowTitle('GUI_TEST')
        self.setWindowIcon(QIcon('test_image.jpg'))
        self.setGeometry(100, 100, 1200, 800)
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyWindow()
    ex.show()
    app.exec_()
