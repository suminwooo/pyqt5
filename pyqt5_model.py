# -*- encoding : utf-8 -*-

import sys

sys.setrecursionlimit(5000)
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
import plotly
import DataAnalysis_
import Track2
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets
import xlrd

track2_result = 0


# 에러로 인한 강제종료 막는 함수
def my_exception_hook(exctype, value, traceback):
    sys._excepthook(exctype, value, traceback)


sys._excepthook = sys.excepthook
sys.excepthook = my_exception_hook


class popup_screen(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def model_result(self, original_box):
        group_box_void = QGroupBox('void')
        group_box_surface = QGroupBox('surface')
        group_box_corona = QGroupBox('corona')
        group_box_noise = QGroupBox('noise')
        void_text = QLabel('num')
        surface_text = QLabel('num')
        corona_text = QLabel('num')
        noise_text = QLabel('num')
        void_detail_layout = QGridLayout()
        void_detail_layout.addWidget(void_text)
        group_box_void.setLayout(void_detail_layout)
        void_layout = QVBoxLayout()
        void_layout.addWidget(group_box_void)
        surface_detail_surface = QGridLayout()
        surface_detail_surface.addWidget(surface_text)
        group_box_surface.setLayout(surface_detail_surface)
        surface_layout = QVBoxLayout()
        surface_layout.addWidget(group_box_surface)
        corona_deail_layout = QGridLayout()
        corona_deail_layout.addWidget(corona_text)
        group_box_corona.setLayout(corona_deail_layout)
        corona_layout = QVBoxLayout()
        corona_layout.addWidget(group_box_corona)
        noise_detail_layout = QGridLayout()
        noise_detail_layout.addWidget(noise_text)
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
        x_max_value = (MyWindow.__dict__['_MyWindow__shared_state']['clustering1'])
        x_min_value = (MyWindow.__dict__['_MyWindow__shared_state']['clustering2'])
        y_max_value = (MyWindow.__dict__['_MyWindow__shared_state']['clustering3'])
        y_min_value = (MyWindow.__dict__['_MyWindow__shared_state']['clustering4'])
        z_max_value = (MyWindow.__dict__['_MyWindow__shared_state']['clustering5'])
        z_min_value = (MyWindow.__dict__['_MyWindow__shared_state']['clustering6'])

        self.value_list = [x_max_value, x_min_value, y_max_value, y_min_value, z_max_value, z_min_value]
        return self.range_labeling.setText(
            '#1 클러스터링 : {}, #2 클러스터링 : {}, #3 클러스터링: {}, #4 클러스터링 : {},#5 클러스터링 : {},'
            ' #6 클러스터링 : {}'.format(
                self.value_list[0], self.value_list[1], self.value_list[2], self.value_list[3], self.value_list[4],
                self.value_list[5]))

    def show_graph_mini(self):
        print('팝업창에서 그래프 표현 한거')
        print(track2_result)

    def setupUI(self):
        # 박스 설정
        self.setWindowTitle("graph detail")
        self.full_group_box = QGroupBox('전체 ')
        self.range_group_box = QGroupBox('범위')
        self.range_box_label = QPushButton('버튼을 눌러서 셋팅한 정보를 확인하세요')
        self.range_labeling = QLabel('버튼을 눌러서 셋팅한 정보를 확인하세요')
        self.range_box_label.clicked.connect(self.range_clicked)
        self.range_box_label.clicked.connect(self.button_block)

        self.graph_box = QGroupBox("그래프")
        self.graph_btn = QPushButton("그래프 생성 버튼")
        self.graph_btn.clicked.connect(self.show_graph_mini)
        self.browser_popup = QtWebEngineWidgets.QWebEngineView()
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
        self.layout_graph_btn.addWidget(self.browser_popup)
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
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        # 박스 설정
        self.graph_box = QGroupBox("Dendrogram")
        self.graph_detail_box = QGroupBox("분류")
        # 레이아웃 설정
        self.layout_all = QVBoxLayout()  # 전체 레이아웃
        self.layout_detail = QVBoxLayout()  # 전체 디테일 레이아웃
        self.range_layout = QVBoxLayout()  # 범위설정 부분
        self.grahp_layout = QVBoxLayout()  # 그래프 부분
        self.graph_box.setLayout(self.grahp_layout)
        self.grahp_layout.addWidget(self.graph_box)
        self.grahp_layout.addWidget(self.graph_detail_box)
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.graph_box, 10)
        self.layout.addWidget(self.graph_detail_box, 5)
        self.setLayout(self.layout)
        self.setGeometry(200, 200, 750, 500) # 사이즈 조정

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
    __shared_state = {"clustering1": '범위X', "clustering2": '범위X', "clustering3": '범위X',
                      "clustering4": '범위X', "clustering5": '범위X', "clustering6": '범위X',
                      "first_axis": '지정축1', "second_axis": '지정축2', "third_axis": '지정축3',
                      }

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
        self.value = DataAnalysis_.Model('data/sample_data/', self.file_name1)
        self.void_text_track1.setText('{} %'.format(float(self.value[0])))
        self.surface_text_track1.setText('{} %'.format(float(self.value[1])))
        self.corona_text_track1.setText('{} %'.format(float(self.value[2])))
        self.noise_text_track1.setText('{} %'.format(float(self.value[3])))

    def popup_btn(self):
        mydialog = popup_screen(self)
        mydialog.show()

    def popup_dendrogram_btn(self):
        mydialog = popup_screen_dendrogram(self)
        mydialog.show()

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
        data_analysis = Track2.Track2_DataAnalysis('data/sample_data', self.track2_file_clicked())
        QMessageBox.about(self, "message", self.file_path2.file_name[-1] + '실행')
        arr = data_analysis.LoadData()
        Denoise_Data = data_analysis.Preprocess(arr)
        self.track2_result = data_analysis.Feature_Extract(Denoise_Data)
        global track2_result
        track2_result = self.track2_result
        self.track2_axis = list(self.track2_result.columns)
        self.combobox_x_track23.addItems(self.track2_axis)
        self.combobox_y_track23.addItems(self.track2_axis)
        self.combobox_z_track23.addItems(['none'] + self.track2_axis)

        return self.track2_result

    def show_graph(self):
        import plotly.io as plt_io
        plt_io.templates["custom_dark"] = plt_io.templates["none"]
        plt_io.templates['custom_dark']['layout']['yaxis']['gridcolor'] = 'darkgray'
        plt_io.templates['custom_dark']['layout']['xaxis']['gridcolor'] = 'darkgray'

        # 그래프 그리기
        x_value = '{}'.format(self.x_content)
        y_value = '{}'.format(self.y_content)
        z_value = '{}'.format(self.z_content)
        self.df = self.track2_file_modeling()

        if z_value == 'none':
            fig = px.scatter(self.df, x=x_value, y=y_value, opacity=0.4)
            fig.layout.template = 'custom_dark'
            fig.update_traces(marker=dict(size=5, line=dict(color='rgba(0, 0, 0, 1)', width=2)),
                              selector=dict(mode='markers'))
            self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))
            return self.df
        else:
            fig = px.scatter_3d(self.df, x=x_value, y=y_value, z=z_value, opacity=0.4)
            fig.layout.template = 'custom_dark'
            fig.update_traces(marker=dict(size=3, line=dict(color='rgba(0, 0, 0, 1)', width=2)),
                              selector=dict(mode='markers'))
            self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))
            return self.df

    def set_axis(self):
        self.x_content = self.combobox_x_track23.currentText()
        self.y_content = self.combobox_y_track23.currentText()
        self.z_content = self.combobox_z_track23.currentText()
        self.change_axis_label_track23.setText('{},{},{}'.format(self.x_content, self.y_content, self.z_content))
        self.__shared_state['first_axis'] = self.x_content
        self.__shared_state['second_axis'] = self.y_content
        self.__shared_state['third_axis'] = self.z_content
        return [self.__shared_state['first_axis'], self.__shared_state['second_axis'],
                self.__shared_state['third_axis']]

    def save_range_value(self):
        self.clustering1 = self.x_max_line.text()
        self.clustering2 = self.x_min_line.text()
        self.clustering3 = self.y_max_line.text()
        self.clustering4 = self.y_min_line.text()
        self.clustering5 = self.z_max_line.text()
        self.clustering6 = self.z_min_line.text()

    def preprocessing_result_track1(self):
        now = time.localtime()
        CREAT_TIME = "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday,
                                                        now.tm_hour, now.tm_min, now.tm_sec)
        svm_df = pd.DataFrame({'FileName': self.file_name1,
                               'CREAT_TIME': CREAT_TIME,
                               'VOID_JDGM': (self.value[3] * 100).round(2),
                               'SURFACE_JDGM': (self.value[2] * 100).round(2),
                               'CORONA_JDGM': (self.value[0] * 100).round(2),
                               'NOISE_JDGM': (self.value[1] * 100).round(2)}, index=[0])
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
        self.dendrogram_btn_track23 = QPushButton('덴드로그램 팝업 버튼') ###########################
        self.dendrogram_btn_track23.clicked.connect(self.popup_dendrogram_btn)
        self.file_path1.selectedItems()
        self.axis_track23 = QGroupBox('클러스터링')
        self.clustering_count_label_track23 = QLabel('클러스터링 입력 수 ')
        self.lineedit_clustering_count_track23 = QLineEdit('0')
        self.center_graph_btn_track23 = QPushButton('그래프 출력 버튼')
        self.graph_axis_track23 = QGroupBox()
        self.center_detail_graph_track23 = QGroupBox('각 축 좌표 범위 검색 및 이미지 출력')
        self.browser = QtWebEngineWidgets.QWebEngineView()
        self.three_graph_track23 = QGroupBox('클러스터링 입력 부분')
        self.search_xyz_track23 = QGroupBox('좌표 검색')
        self.x_max_track23 = QLabel('#1')
        self.x_max_value_track23 = QLineEdit('')
        self.x_min_track23 = QLabel('#4')
        self.x_min_value_track23 = QLineEdit('')
        self.y_max_track23 = QLabel('#2')
        self.y_max_value_track23 = QLineEdit('')
        self.y_min_track23 = QLabel('#5')
        self.y_min_value_track23 = QLineEdit('')
        self.z_max_track23 = QLabel('#3')
        self.z_max_value_track23 = QLineEdit('')
        self.z_min_track23 = QLabel('#6')
        self.z_min_value_track23 = QLineEdit('')
        self.axis_setting_btn_track23 = QPushButton('클러스터링 입력')
        self.axis_setting_btn_track23.clicked.connect(self.check_clicked_range)
        self.axis_setting_btn_track23.clicked.connect(self.save_range_value)
        self.x_max_line = self.x_max_value_track23
        self.x_min_line = self.x_min_value_track23
        self.y_max_line = self.y_max_value_track23
        self.y_min_line = self.y_min_value_track23
        self.z_max_line = self.z_max_value_track23
        self.z_min_line = self.z_min_value_track23
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
        self.layout_axis_track23 = QGridLayout()  # 축선택 레이아웃
        self.layout_axis_track23.addWidget(self.clustering_count_label_track23,0,0)
        self.layout_axis_track23.addWidget(self.lineedit_clustering_count_track23,0,1)
        # self.layout_axis_track23.addWidget(self.center_graph_label_track23,1,0)
        self.layout_axis_track23.addWidget(self.center_graph_btn_track23,1,1)
        # self.layout_axis_track23.addWidget(self.axis_x_axis_track23, 0, 0, 1, 1)
        # self.layout_axis_track23.addWidget(self.combobox_x_track23, 0, 1, 1, 2)
        # self.layout_axis_track23.addWidget(self.axis_labeling_track23, 0, 4, 1, 3)
        # self.layout_axis_track23.addWidget(self.axis_y_axis_track23, 1, 0, 1, 1)
        # self.layout_axis_track23.addWidget(self.combobox_y_track23, 1, 1, 1, 2)
        # self.layout_axis_track23.addWidget(self.change_axis_label_track23, 1, 4, 1, 3)
        # self.layout_axis_track23.addWidget(self.axis_z_axis_track23, 2, 0, 1, 1)
        # self.layout_axis_track23.addWidget(self.combobox_z_track23, 2, 1, 1, 2)
        # self.layout_axis_track23.addWidget(self.axis_z_axis_value_track23, 2, 4, 1, 3)
        self.axis_track23.setLayout(self.layout_axis_track23)
        self.layout_second_track23.addWidget(self.csv_group_box_drop_track23, 2)
        self.layout_second_track23.addWidget(self.csv_group_check_btn_track23, 1)
        self.layout_second_track23.addWidget(self.dendrogram_btn_track23, 1)
        self.layout_second_track23.addWidget(self.axis_track23, 10)
        self.csv_group_box_track23.setLayout(self.layout_second_track23)
        self.layout_second_track23 = QHBoxLayout()  # 오른쪽 두번째 레이아웃
        self.layout_second_graph_btn_track23 = QVBoxLayout()  # 오른쪽 두번째 그래프 부분 레이아웃
        # self.layout_second_graph_btn_track23.addWidget(self.center_graph_btn_track23, alignment=QtCore.Qt.AlignLeft)
        self.layout_second_graph_btn_track23.addWidget(self.browser)
        self.graph_axis_track23.setLayout(self.layout_second_graph_btn_track23)
        self.graph_axis_track23.setLayout(self.layout_second_track23)
        self.layout_second_track23.addWidget(self.graph_axis_track23)
        self.layout_third_track23 = QHBoxLayout()  # 오른쪽 세번째 레이아웃
        self.layout_third_3_detail_1_track23 = QGridLayout()
        self.layout_third_3_detail_1_track23.addWidget(self.x_max_track23, 0, 0)
        self.layout_third_3_detail_1_track23.addWidget(self.x_max_value_track23, 0, 1)
        self.layout_third_3_detail_1_track23.addWidget(self.y_max_track23, 0, 2)
        self.layout_third_3_detail_1_track23.addWidget(self.x_min_value_track23, 0, 3)
        self.layout_third_3_detail_1_track23.addWidget(self.z_max_track23, 0, 4)
        self.layout_third_3_detail_1_track23.addWidget(self.y_max_value_track23, 0, 5)
        self.layout_third_3_detail_1_track23.addWidget(self.x_min_track23, 1, 0)
        self.layout_third_3_detail_1_track23.addWidget(self.y_min_value_track23, 1, 1)
        self.layout_third_3_detail_1_track23.addWidget(self.y_min_track23, 1, 2)
        self.layout_third_3_detail_1_track23.addWidget(self.z_max_value_track23, 1, 3)
        self.layout_third_3_detail_1_track23.addWidget(self.z_min_track23, 1, 4)
        self.layout_third_3_detail_1_track23.addWidget(self.z_min_value_track23, 1, 5)
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
