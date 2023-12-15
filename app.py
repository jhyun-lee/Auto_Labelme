# -*- coding: utf-8 -*-

import functools
import html
import math
import os
import os.path as osp
import re
import webbrowser
import sys
import cv2
import json


import imgviz
import natsort
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets
from PyQt5.QtWidgets import QCheckBox,QComboBox,QVBoxLayout,QLabel,QTabWidget


import pose_detect

import model as model

import shutil

from labelme import __appname__
from labelme import PY2

import utils
import yaml

from config import get_config
from labelme.label_file import LabelFile
from labelme.label_file import LabelFileError
from labelme.logger import logger
from shape import Shape
from labelme.widgets import BrightnessContrastDialog
from widgets import Canvas
from labelme.widgets import FileDialogPreview
from labelme.widgets import LabelDialog
from labelme.widgets import LabelListWidget
from labelme.widgets import LabelListWidgetItem
from labelme.widgets import ToolBar
from labelme.widgets import UniqueLabelQListWidget
from labelme.widgets import ZoomWidget

# FIXME
# - [medium] Set max zoom value to something big enough for FitWidth/Window

# TODO(unknown):
# - Zoom is too "steppy".


LABEL_COLORMAP = imgviz.label_colormap()


class MainWindow(QtWidgets.QMainWindow):

    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(  ## 일부 값들 초기화
        self,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        
        self.selectPersonCombo="P1"  
        labels = []

        if output is not None:
            logger.warning(
                "argument output is deprecated, use output_file instead"
            )
            if output_file is None:
                output_file = output

        

        config=None
        if config is None:  ## 단축키 및 설정값 불러오기  >>  default_config.yaml 파일에서 설정  ************
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "config/default_config.yaml")
            with open(config_path, encoding="UTF-8") as f:
                _cfg = yaml.safe_load(f)

            config = _cfg
        

        self._config = config

        with open("Predefined/PersonDetect.txt", "r") as file:  ## 미리 정의해둘 라벨 이름들 추가-------------------------------------------------------------
            for line in file:
                label = line.strip()  # 줄바꿈 문자 제거
                
                labels.append(label)

        self._config["labels"]=labels


        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]

        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Whether we need to save or not.
        self.dirty = False
        self._noSelectionSlot = False
        self._copied_shapes = None

        # Main widgets and related state.
        self.labelDialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self._config["label_flags"],
        )

        self.labelList = LabelListWidget()
        self.lastOpenDir = None


        ## ---- 비디오 관련 변수들 
        self.vidcap=None   # -------------------- 영상 변수

        #-------------------------- 설정값-------------------------
        self.fps = 1  ################ --- 단위 프레임  기본 10
        self.Addcut=10  ###############  --- 한번에 저장할 사진수 
        self.jpeg_quality = 100   ### 사진의 퀄리티  0~100    100에 가까울수록 사진퀄리티 업 용량 업



        self.length = 0
        self.img_count = 0
        self.vid_count=0
        self.video_list=[]
        self.video_mode=False
        self.video_path=None # 영상 위치  끝 4자리 빼고 _js가 파일이름 +"/프레임번호" 불러오기
        

        ## - ----------------------------------라벨 리스트

        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        self.labelList.itemChanged.connect(self.labelItemChanged)
        self.labelList.itemDropped.connect(self.labelOrderChanged)
        self.shape_dock=QtWidgets.QDockWidget(
            self.tr("Polygon Labels"), self
        )

        self.shape_dock.setObjectName("Labels")
        self.shape_dock.setWidget(self.labelList)


        ## --------------------------------- 라벨 종류 리스트
        self.uniqLabelList = UniqueLabelQListWidget()
        self.uniqLabelList.setToolTip(
            self.tr(
                "Select label to start annotating for it. "
                "Press 'Esc' to deselect."
            )
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                
                
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)


        

        self.label_dock = QtWidgets.QDockWidget(self.tr("Label List"), self)
        self.label_dock.setObjectName("Label List")
        self.label_dock.setWidget(self.uniqLabelList)



        #-------------------------옵션값-------------
        
        self.tabs = QTabWidget()

        self.autoButton_person=QtWidgets.QPushButton('현재 이미지\n 오토라벨링(사람)',self)
        self.autoButton_person.clicked.connect(self.AutoBounding_person)

        self.autoButton_handBox=QtWidgets.QPushButton('현재 이미지\n 오토라벨링(핸드박스)',self)
        self.autoButton_handBox.clicked.connect(self.AutoBounding_handBox)

        self.autoButton_PersonAndhand=QtWidgets.QPushButton('현재 이미지\n 오토라벨링(사람 손)',self)
        self.autoButton_PersonAndhand.clicked.connect(self.AutoBounding_PersonAndhand)
        

        self.autoButton_hand=QtWidgets.QPushButton('현재 이미지\n 오토라벨링(손)',self)
        self.autoButton_hand.clicked.connect(self.AutoBounding_hand)


        # 오토라벨링 현재이미지 --------------------------------------------
        layout_1 = QtWidgets.QHBoxLayout()

        layout_1.addWidget(self.autoButton_person)
        layout_1.addWidget(self.autoButton_handBox)
        layout_1.addWidget(self.autoButton_PersonAndhand)
        layout_1.addWidget(self.autoButton_hand)

        
        container_widget_1 = QtWidgets.QWidget()
        container_widget_1.setLayout(layout_1)


        # self.Option_dock = QtWidgets.QDockWidget(self.tr("Auto Labeling"), self)
        # self.Option_dock.setObjectName("Auto Labeling")
        # self.Option_dock.setWidget(container_widget_1)



        self.auto_AllButton_person=QtWidgets.QPushButton('모든 이미지\n 오토라벨링(사람)',self)
        self.auto_AllButton_person.clicked.connect(self.All_autolabeling_person)

        self.auto_AllButton_handBox=QtWidgets.QPushButton('모든 이미지\n 오토라벨링(핸드박스)',self)
        self.auto_AllButton_handBox.clicked.connect(self.All_autolabeling_handBox)

        self.auto_AllButton_PersonAndhand=QtWidgets.QPushButton('모든 이미지\n 오토라벨링(사람 손)',self)
        self.auto_AllButton_PersonAndhand.clicked.connect(self.All_autolabeling_PersonAndhand)
        

        self.auto_AllButton_hand=QtWidgets.QPushButton('모든 이미지\n 오토라벨링(손)',self)
        self.auto_AllButton_hand.clicked.connect(self.All_autolabeling_hand)



        # 오토라벨링 전체 --------------------------------------------
        layout_2 = QtWidgets.QHBoxLayout()

        layout_2.addWidget(self.auto_AllButton_person)
        layout_2.addWidget(self.auto_AllButton_handBox)
        layout_2.addWidget(self.auto_AllButton_PersonAndhand)
        layout_2.addWidget(self.auto_AllButton_hand)

        container_widget_2 = QtWidgets.QWidget()
        container_widget_2.setLayout(layout_2)
        

        # self.Labeling_all_dock = QtWidgets.QDockWidget(self.tr("Auto Labeling All"), self)
        # self.Labeling_all_dock.setObjectName("Auto Labeling All")
        # self.Labeling_all_dock.setWidget(container_widget_2)



        ## 전신 스켈레톤 관련 ----------------------------------------------

        self.auto_person_Point=QtWidgets.QPushButton('현재 이미지\n 전신 라벨링(사람)',self)
        self.auto_person_Point.clicked.connect(self.AutoBounding_personSkel)

        self.auto_person_Skel=QtWidgets.QPushButton('현재 이미지\n 전신 구성(관절 포인트 기반)',self)
        self.auto_person_Skel.clicked.connect(self.AutoBounding_personSkel_Body)

        self.auto_person_Skel_remove=QtWidgets.QPushButton('현재 이미지\n 전신 스켈레톤 삭제',self)
        self.auto_person_Skel_remove.clicked.connect(self.AutoBounding_personSkel_remove)

        layout_3 = QtWidgets.QHBoxLayout()

        layout_3.addWidget(self.auto_person_Point)
        layout_3.addWidget(self.auto_person_Skel)
        layout_3.addWidget(self.auto_person_Skel_remove)
        
       

        container_widget_3 = QtWidgets.QWidget()
        container_widget_3.setLayout(layout_3)

        ## 전신 스켈레톤 모든 이미지 -------------------------------------------------

        self.auto_person_Point_all=QtWidgets.QPushButton('전체 이미지\n 전신 라벨링(사람)',self)
        self.auto_person_Point_all.clicked.connect(self.All_AutoBounding_personSkel)

        self.auto_person_Skel_all=QtWidgets.QPushButton('전체 이미지\n 전신 구성(관절 포인트 기반)',self)
        self.auto_person_Skel_all.clicked.connect(self.All_AutoBounding_personSkel_Body)


        layout_4 = QtWidgets.QHBoxLayout()

        layout_4.addWidget(self.auto_person_Point_all)
        layout_4.addWidget(self.auto_person_Skel_all)

        
        container_widget_4 = QtWidgets.QWidget()
        container_widget_4.setLayout(layout_4)




        self.tabs.addTab(container_widget_1, 'Auto Labeling')
        self.tabs.addTab(container_widget_2, 'Auto Labeling All')
        self.tabs.addTab(container_widget_3, 'Auto BodySkeleton')
        self.tabs.addTab(container_widget_4, 'Auto All BodySkeleton')


        vbox = QVBoxLayout()
        vbox.addWidget(self.tabs)

        container_widget_tab = QtWidgets.QWidget()
        container_widget_tab.setLayout(vbox)


        self.Autotabs_dock = QtWidgets.QDockWidget(self.tr("Auto Tabs"), self)
        self.Autotabs_dock.setObjectName("Auto Tabs")

        
        self.Autotabs_dock.setWidget(container_widget_tab)

        

        #----------------------------------- 콤보 박스
        
        persontList=["P1","P2","P3","P5","P6","P7","D"]
        HandList=["RL","RH","LH"]
        fingerList =['Thum','Index','Middle','Ring','Pinky']

        

        self.combo = QComboBox(self)

        for person in persontList:
            self.combo.addItem(person)
        
        self.combo.activated[str].connect(self.onActivated) ## 선택하고 


        self.auto_hand_change=QtWidgets.QPushButton('손 방향 변경',self)
        self.auto_hand_change.clicked.connect(self.handChange)

        self.PersonReDetect=QtWidgets.QPushButton('사람 재탐색',self)
        self.PersonReDetect.clicked.connect(self.OnePerson)


        self.PreLabel=QtWidgets.QPushButton('앞선 라벨링불러오기',self)
        self.PreLabel.clicked.connect(self.PreLabel_F)
        


        self.precombo1 = QComboBox(self)
        self.precombo2 = QComboBox(self)
        

        for person in persontList:
            self.precombo1.addItem(person)

        for Hand in HandList:
            self.precombo2.addItem(Hand)


        

        #----------------------------------- 콤보 박스

        # 오토라벨링 OPT --------------------------------------------
        layout_3_1 = QtWidgets.QHBoxLayout()
        layout_3_1.addWidget(self.auto_hand_change)
        layout_3_1.addWidget(self.combo)

        layout_3_2 = QtWidgets.QHBoxLayout()
        layout_3_2.addWidget(self.PersonReDetect)


        layout_3_3 = QtWidgets.QHBoxLayout()
        layout_3_3.addWidget(self.PreLabel)
        layout_3_3.addWidget(self.precombo1)
        layout_3_3.addWidget(self.precombo2)



        container_widget_3 = QtWidgets.QWidget()

        # 수직 방향 레이아웃을 생성하고 container_widget_2를 추가
        container_layout = QtWidgets.QVBoxLayout(container_widget_3)
        container_layout.addLayout(layout_3_1)
        container_layout.addLayout(layout_3_2)
        container_layout.addLayout(layout_3_3)
        

        self.Labeling_opt_dock = QtWidgets.QDockWidget(self.tr("Auto Labeling opt"), self)
        self.Labeling_opt_dock.setObjectName("Auto Labeling opt")

        # container_widget_2를 self.Labeling_opt_dock의 위젯으로 설정
        self.Labeling_opt_dock.setWidget(container_widget_3)



        # 액션 라벨링 --------------------------------------------

        self.actionList=["일반","카드/손","칩/손","물건/손","얼굴/손","특수/손","테이블 이탈"]
        

        self.ActionComboList=[]

        for i in range(0,14):
            actioncombo=QComboBox(self)
            actioncombo.setFixedHeight(25)
            self.ActionComboList.append(actioncombo)

        for item in self.actionList:
            for combo in self.ActionComboList:
                combo.addItem(item)
           
            
            
        
        #self.combo.activated[str].connect(self.onActivated) ## 선택하고 


        self.ActionLabelingbtn=QtWidgets.QPushButton('액션 라벨링',self)
        self.ActionLabelingbtn.clicked.connect(self.ActionLabeling)


        layout_4_Last = QtWidgets.QHBoxLayout()
        layout_4_Last.addWidget(self.ActionLabelingbtn)

        container_widget_4 = QtWidgets.QWidget()

        
        container_layout_4 = QtWidgets.QVBoxLayout(container_widget_4)


        layout_4_start = QtWidgets.QHBoxLayout()
        label1= QLabel("사람")
        label2= QLabel("오른손(Base)")
        label3= QLabel("왼손")

        layout_4_start.addWidget(label1)
        layout_4_start.addWidget(label2)
        layout_4_start.addWidget(label3)

        container_layout_4.addLayout(layout_4_start)

        index=0
        for Person in persontList:
            
            label= QLabel(Person)

            layout_4_1 = QtWidgets.QHBoxLayout()
            layout_4_1.addWidget(label)
            layout_4_1.addWidget(self.ActionComboList[index])
            layout_4_1.addWidget(self.ActionComboList[index+1])


            container_layout_4.addLayout(layout_4_1)
            index+=2



        container_layout_4.addLayout(layout_4_Last)
        
        

        self.Labeling_Action_dock = QtWidgets.QDockWidget(self.tr("Action Labeling"), self)
        self.Labeling_Action_dock.setObjectName("Action Labeling")

        
        self.Labeling_Action_dock.setWidget(container_widget_4)


        




        ## -------------------------------- 파일 리스트
        self.fileSearch = QtWidgets.QLineEdit()
        self.fileSearch.setPlaceholderText(self.tr("Search Filename"))



        self.fileSearch.textChanged.connect(self.fileSearchChanged)
        self.fileListWidget = QtWidgets.QListWidget()
        self.fileListWidget.itemSelectionChanged.connect(
            self.fileSelectionChanged
        )
        
        fileListLayout = QtWidgets.QVBoxLayout()
        fileListLayout.setContentsMargins(0, 0, 0, 0)
        fileListLayout.setSpacing(0)
        fileListLayout.addWidget(self.fileSearch)
        fileListLayout.addWidget(self.fileListWidget)

        self.file_dock = QtWidgets.QDockWidget(self.tr("File List"), self)
        self.file_dock.setObjectName("Files")
        fileListWidget = QtWidgets.QWidget()
        fileListWidget.setLayout(fileListLayout)
        self.file_dock.setWidget(fileListWidget)

        self.zoomWidget = ZoomWidget()
        self.setAcceptDrops(True)

        ## --------------------------------------------------------------
        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scrollArea)

        

        features=(QtWidgets.QDockWidget.NoDockWidgetFeatures)

        #self.Autotabs_dock.show()
        
        ## 닫기 버튼 비활성화
        # for dock in [ "Autotabs_dock","Labeling_opt_dock","Labeling_Action_dock","shape_dock", "file_dock"]:# ,"label_dock"
        #     getattr(self, dock).setFeatures(features)
        # ,"Option_dock","Labeling_all_dock"
        #  


        self.addDockWidget(Qt.RightDockWidgetArea, self.Autotabs_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.Labeling_opt_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.Labeling_Action_dock)
        #self.addDockWidget(Qt.RightDockWidgetArea, self.flag_dock)
        #self.addDockWidget(Qt.RightDockWidgetArea, self.Option_dock)
        #self.addDockWidget(Qt.RightDockWidgetArea, self.Labeling_all_dock)


        #self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)
        

        
        


        # Actions
        action = functools.partial(utils.newAction, self)
        shortcuts = self._config["shortcuts"]
        quit = action(
            self.tr("&Quit" + str(shortcuts["quit"])),
            self.close,
            shortcuts["quit"],
            "quit",
            self.tr("Quit application"),
        )
        open_ = action(
            self.tr("&Open"),
            self.openFile,
            shortcuts["open"],
            "open",
            self.tr("Open image or label file"),
        )

        opendir = action(
            self.tr("&Open Dir\n("+ str(shortcuts["open_dir"]+")")),
            self.openDirDialog,
            shortcuts["open_dir"],
            "open",
            self.tr("Open Dir"),
        )


        openNextImg = action(
            self.tr("&Next Image\n("+ str(shortcuts["open_next"]+")")),
            self.openNextImg,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        openPrevImg = action(
            self.tr("&Prev Image\n("+ str(shortcuts["open_prev"]+")")),
            self.openPrevImg,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctl+Shift to copy labels)"),
            enabled=False,
        )
        save = action(
            self.tr("&Save\n("+ str(shortcuts["save"]+")")),
            self.saveFile,
            shortcuts["save"],
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )
        saveAs = action(
            self.tr("&Save As"),
            self.saveFileAs,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=False,
        )

        deleteFile = action(
            self.tr("&Delete File"),
            self.deleteFile,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=False,
        )

        changeOutputDir = action(
            self.tr("&Change Output Dir"),
            slot=self.changeOutputDirDialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        saveAuto = action(
            text=self.tr("Save &Automatically"),
            slot=lambda x: self.actions.saveAuto.setChecked(x),
            icon="save",
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
        )
        saveAuto.setChecked(self._config["auto_save"])

        saveWithImageData = action(
            text="Save With Image Data",
            slot=self.enableSaveImageWithData,
            tip="Save image data in label file",
            checkable=True,
            checked=self._config["store_data"],
        )

        close = action(
            "&Close",
            self.closeFile,
            shortcuts["close"],
            "close",
            "Close current file",
        )

        toggle_keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            self.toggleKeepPrevMode,
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "keep pevious annotation" mode'),
            checkable=True,
        )
        toggle_keep_prev_mode.setChecked(self._config["keep_prev"])

        createMode = action(  ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 다각형
            self.tr("Create Polygons\n("+ str(shortcuts["create_polygon"]+")")),
            lambda: self.toggleDrawMode(False, createMode="polygon"),
            shortcuts["create_polygon"],
            "objects",
            self.tr("Start drawing polygons"),
            enabled=False,
        )

        createRectangleMode = action(## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 사각형
            self.tr("Create Rectangle\n("+ str(shortcuts["create_rectangle"]+")")),
            lambda: self.toggleDrawMode(False, createMode="rectangle"),
            shortcuts["create_rectangle"],
            "objects",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )

        createCircleMode = action(
            self.tr("Create Circle"),
            lambda: self.toggleDrawMode(False, createMode="circle"),
            shortcuts["create_circle"],
            "objects",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        createLineMode = action(
            self.tr("Create Line\n("+ str(shortcuts["create_line"]+")")),
            lambda: self.toggleDrawMode(False, createMode="line"),
            shortcuts["create_line"],
            "objects",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        createPointMode = action(
            self.tr("Create Point\n("+ str(shortcuts["create_point"]+")")),
            lambda: self.toggleDrawMode(False, createMode="point"),
            shortcuts["create_point"],
            "objects",
            self.tr("Start drawing points"),
            enabled=False,
        )
        createLineStripMode = action(
            self.tr("Create LineStrip"),
            lambda: self.toggleDrawMode(False, createMode="linestrip"),
            shortcuts["create_linestrip"],
            "objects",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        createAiPolygonMode = action(
            self.tr("Create AI-Polygon"),
            lambda: self.toggleDrawMode(False, createMode="ai_polygon"),
            None,
            "objects",
            self.tr("Start drawing ai_polygon. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        editMode = action(
            self.tr("Edit Polygons\n("+ str(shortcuts["edit_polygon"]+")")),
            self.setEditMode,
            shortcuts["edit_polygon"],
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )

        delete = action(
            self.tr("Delete Polygons\n("+ str(shortcuts["delete_polygon"]+")")),
            self.deleteSelectedShape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Polygons\n("+ str(shortcuts["copy_polygon"]+")")),
            self.copySelectedShape,
            shortcuts["copy_polygon"],
            "copy_clipboard",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Polygons\n("+ str(shortcuts["paste_polygon"]+")")),
            self.pasteSelectedShape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=False,
        )
        findUnlabeledFile = action(
            self.tr("Find Unlabeled File\n("+ str(shortcuts["find_unlabeled_file"]+")")),
            self.findUnlabeledFile,
            shortcuts["find_unlabeled_file"],
            "None",
            self.tr("Find Unlabeled File"),
            enabled=False,
        )
        undoLastPoint = action(
            self.tr("Undo last point\n("+ str(shortcuts["undo_last_point"]+")")),
            self.canvas.undoLastPoint,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )







        removePoint = action(
            text="Remove Selected Point",
            slot=self.removeSelectedPoint,
            shortcut=shortcuts["remove_selected_point"],
            icon="edit",
            tip="Remove selected point from polygon",
            enabled=False,
        )

        undo = action(
            self.tr("Undo\n("+ str(shortcuts["undo"]+")")),
            self.undoShapeEdit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )











        hideAll = action(
            self.tr("&Hide\nPolygons"),
            functools.partial(self.togglePolygons, False),
            icon="eye",
            tip=self.tr("Hide all polygons"),
            enabled=False,
        )
        showAll = action(
            self.tr("&Show\nPolygons"),
            functools.partial(self.togglePolygons, True),
            icon="eye",
            tip=self.tr("Show all polygons"),
            enabled=False,
        )



        zoom = QtWidgets.QWidgetAction(self)
        zoomBoxLayout = QtWidgets.QVBoxLayout()
        zoomBoxLayout.addWidget(self.zoomWidget)
        zoomLabel = QtWidgets.QLabel("Zoom")
        zoomLabel.setAlignment(Qt.AlignCenter)
        zoomLabel.setFont(QtGui.QFont(None, 10))
        zoomBoxLayout.addWidget(zoomLabel)
        zoom.setDefaultWidget(QtWidgets.QWidget())
        zoom.defaultWidget().setLayout(zoomBoxLayout)
        self.zoomWidget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmtShortcut(
                    "{},{}".format(shortcuts["zoom_in"], shortcuts["zoom_out"])
                ),
                utils.fmtShortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoomWidget.setEnabled(False)

        zoomIn = action(
            self.tr("Zoom &In"),
            functools.partial(self.addZoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoomOut = action(
            self.tr("&Zoom Out"),
            functools.partial(self.addZoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoomOrg = action(
            self.tr("&Original size"),
            functools.partial(self.setZoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keepPrevScale = action(
            self.tr("&Keep Previous Scale"),
            self.enableKeepPrevScale,
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        fitWindow = action(
            self.tr("&Fit Window"),
            self.setFitWindow,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fitWidth = action(
            self.tr("Fit &Width"),
            self.setFitWidth,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightnessContrast = action(
            "&Brightness Contrast",
            self.brightnessContrast,
            None,
            "color",
            "Adjust brightness and contrast",
            enabled=False,
        )
        # Group zoom controls into a list for easier toggling.
        zoomActions = (
            self.zoomWidget,
            zoomIn,
            zoomOut,
            zoomOrg,
            fitWindow,
            fitWidth,
        )
        self.zoomMode = self.FIT_WINDOW
        fitWindow.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("&Edit Label"),
            self.editLabel,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.setFillDrawing,
            None,
            "color",
            self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        if self._config["canvas"]["fill_drawing"]:
            fill_drawing.trigger()

        # Lavel list context menu.
        labelMenu = QtWidgets.QMenu()
        utils.addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu
        )

        ## -------------------------------- -Load_Video- -------------------------------- 버튼 추가


        LoadVideo = action(
            self.tr("Load Video\n("+ str(shortcuts["Load_video"]+")")),## 이름
            self.Load_Video_save,## 함수
            shortcuts["Load_video"],## 단축키
            "LoadVideo",  ## >>> 이게 멀까
            self.tr("Load Video"),## 툴팁
        )


        

        ## -------------------------------- -All load- ----------------------------

        All_load = action(
            self.tr("All load\n("+ str(shortcuts["all_load"]+")")),## 이름
            self.All_load,## 함수
            shortcuts["all_load"],## 단축키
            "all_load",  ## >>> 이게 멀까
            self.tr("All load"),## 툴팁
            enabled=False, ## 
        )

         ## -------------------------------- -Prev json- ----------------------------
        prev_img_bounding = action(
            self.tr("Prev Img Bounding\n("+ str(shortcuts["prev_img_bounding"]+")")),## 이름
            self.prev_img_bounding,## 함수
            shortcuts["prev_img_bounding"],## 단축키
            "prev_img_bounding",  ## >>> 이게 멀까
            self.tr("prev bounding Load"),## 툴팁
            enabled=False, ## 
        )


        ## -------------------------------- -Polygon  To Rectagle- ----------------------------

        # Convert_to_Rectangle = action(
        #     self.tr("Convert all to Rectangle\n("+ str(shortcuts["To_Rectagle"]+")")),## 이름
        #     self.PolygonToRectangle,## 함수
        #     shortcuts["To_Rectagle"],## 단축키
        #     "Convert_to_Rectangle",  ## >>> 이게 멀까
        #     self.tr("Convert all to Rectangle"),## 툴팁
        #     enabled=False, ## 
        # )

         ## -------------------------------- -Auto_Bounding- ---------------------------- 기능 추가

        Auto_Bounding = action(
            self.tr("Auto Bounding\n("+ str(shortcuts["Auto_Bounding"]+")")),## 이름
            self.AutoBounding_person,## 함수
            shortcuts["Auto_Bounding"],## 단축키
            "Auto_Bounding",  ## >>> 이게 멀까
            self.tr("Auto_Bounding"),## 툴팁
            enabled=False, ## 
        )

        self.auto_box=False
        self.auto_label_Text=''

        # Store actions for further handling.
        self.actions = utils.struct(  ## 모든 기능들
            saveAuto=saveAuto,
            saveWithImageData=saveWithImageData,
            changeOutputDir=changeOutputDir,
            save=save,
            saveAs=saveAs,
            open=open_,
            close=close,
            deleteFile=deleteFile,
            toggleKeepPrevMode=toggle_keep_prev_mode,
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            copy=copy,
            paste=paste,
            undoLastPoint=undoLastPoint,
            undo=undo,
            #findUnlabeledFile=findUnlabeledFile,
            removePoint=removePoint,
            createMode=createMode,
            editMode=editMode,
            createRectangleMode=createRectangleMode,
            createCircleMode=createCircleMode,
            createLineMode=createLineMode,
            createPointMode=createPointMode,
            createLineStripMode=createLineStripMode,
            createAiPolygonMode=createAiPolygonMode,
            zoom=zoom,
            zoomIn=zoomIn,
            zoomOut=zoomOut,
            zoomOrg=zoomOrg,
            keepPrevScale=keepPrevScale,
            fitWindow=fitWindow,
            fitWidth=fitWidth,
            brightnessContrast=brightnessContrast,
            zoomActions=zoomActions,
            openNextImg=openNextImg,
            openPrevImg=openPrevImg,

            All_load=All_load,
            LoadVideo=LoadVideo,
            #Auto_Bounding=Auto_Bounding,
            prev_img_bounding=prev_img_bounding,
            #Convert_to_Rectangle=Convert_to_Rectangle,
            fileMenuActions=(open_, opendir, save, saveAs, close, quit),
            tool=(),
            # XXX: need to add some actions here to activate the shortcut
            # menu shown at right click  
            menu=(# 우클릭시 메뉴
                createMode,
                createRectangleMode,
                # createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                # createAiPolygonMode,
                # editMode,
                edit,
                # duplicate,
                # copy,
                # paste,
                delete,
                undo,
                undoLastPoint,
                removePoint,
            ),
            onLoadActive=(
                close,
                createMode,
                createRectangleMode,
                createCircleMode,
                createLineMode,
                createPointMode,
                createLineStripMode,
                createAiPolygonMode,
                editMode,
                brightnessContrast,
            ),
            onShapesPresent=(saveAs, hideAll, showAll),
        )

        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        self.menus = utils.struct( ## 전체 메뉴
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),  ## 토글메뉴 껏을때
            # help=self.menu(self.tr("&Help")),
            recentFiles=QtWidgets.QMenu(self.tr("Open &Recent")),
            labelList=labelMenu,
        )

        utils.addActions( ## File 메뉴
            self.menus.file,
            (
                openNextImg,
                openPrevImg,
                opendir,
                save,
                None,
                close,
                quit,
            ),
        )

        utils.addActions(
            self.menus.view,
            (
                self.Autotabs_dock.toggleViewAction(),
                self.Labeling_opt_dock.toggleViewAction(),
                self.Labeling_Action_dock.toggleViewAction(),
                
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
              
            ),
        )

        
        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        utils.addActions(self.canvas.menus[0], self.actions.menu)
        utils.addActions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copyShape),
                action("&Move here", self.moveShape),
            ),
        )

        selectAiModel = QtWidgets.QWidgetAction(self)
        selectAiModel.setDefaultWidget(QtWidgets.QWidget())
        selectAiModel.defaultWidget().setLayout(QtWidgets.QVBoxLayout())
        self._selectAiModelComboBox = QtWidgets.QComboBox()
        selectAiModel.defaultWidget().layout().addWidget(
            self._selectAiModelComboBox
        )

        self._selectAiModelComboBox.setCurrentIndex(1)
        self._selectAiModelComboBox.setEnabled(False)
        self._selectAiModelComboBox.currentIndexChanged.connect(
            lambda: self.canvas.initializeAiModel(
                name=self._selectAiModelComboBox.currentText()
            )
        )
        selectAiModelLabel = QtWidgets.QLabel(self.tr("AI Model"))
        selectAiModelLabel.setAlignment(QtCore.Qt.AlignCenter)
        selectAiModelLabel.setFont(QtGui.QFont(None, 10))
        selectAiModel.defaultWidget().layout().addWidget(selectAiModelLabel)

        self.tools = self.toolbar("Tools")
        self.actions.tool = (## 좌측 툴바
            opendir,
            LoadVideo,
            None,
            save,
            openPrevImg,
            openNextImg,
            None,
            createMode,
            createRectangleMode,
            createLineMode,
            createPointMode,
            createLineStripMode,
            editMode,
            # copy,
            # paste,
            # delete,
            None,
            # --------------------------- 기능추가 위치_------------------------------------
            All_load,
            prev_img_bounding,
            #Convert_to_Rectangle,
            None,
            #findUnlabeledFile,
            #Auto_Bounding,
            undo,
        )

        self.statusBar().showMessage(str(self.tr("%s started.")) % __appname__)
        self.statusBar().show()

        if output_file is not None and self._config["auto_save"]:
            logger.warn(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.imagePath = None
        self.recentFiles = []
        self.maxRecent = 7
        self.otherData = None
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightnessContrast_values = {}
        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        if filename is not None and osp.isdir(filename):
            self.importDirImages(filename, load=False)
        else:
            self.filename = filename

        if config["file_search"]:
            self.fileSearch.setText(config["file_search"])
            self.fileSearchChanged()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("labelme", "labelme")
        self.recentFiles = self.settings.value("recentFiles", []) or []
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(state)

        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queueEvent(functools.partial(self.loadFile, self.filename))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # self.firstStart = True
        # if self.firstStart:
        #    QWhatsThis.enterWhatsThisMode()


    def imwriteKor(self,filename, img, params=None):
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
        
    


    def onActivated(self,str):## 콤보 박스
           
            self.selectPersonCombo=str

            


    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            utils.addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        return toolbar

    # Support Functions

    def noShapes(self):
        return not len(self.labelList)

    def populateModeActions(self):  ## edit 요소 수정 ! 부분
        tool, menu = self.actions.tool, self.actions.menu
        self.tools.clear()
        utils.addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createLineMode,
            self.actions.createLineStripMode,
            self.actions.createPointMode,
            self.actions.editMode,
        )
        editMenu=(
                self.actions.edit,
                self.actions.duplicate,
                self.actions.delete,
                None,
                self.actions.undo,
                None,
                self.actions.removePoint,

            )
        
        utils.addActions(self.menus.edit, actions + editMenu)  ## edit메뉴 

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)

        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            label_file = osp.splitext(self.imagePath)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            self.saveLabels(label_file)
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = "{} - {}*".format(title, self.filename)
        self.setWindowTitle(title)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createCircleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createLineStripMode.setEnabled(True)
        self.actions.createAiPolygonMode.setEnabled(True)



        title = __appname__
        if self.filename is not None:
            title = "{} - {}".format(title, self.filename)
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self.otherData = None
        self.canvas.resetState()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # Callbacks

    def undoShapeEdit(self):
        self.canvas.restoreShape()
        self.labelList.clear()
        self.loadShapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)



    def toggleDrawingSensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.editMode.setEnabled(not drawing)
        self.actions.undoLastPoint.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            self.actions.createMode.setEnabled(True)
            self.actions.createRectangleMode.setEnabled(True)
            self.actions.createCircleMode.setEnabled(True)
            self.actions.createLineMode.setEnabled(True)
            self.actions.createPointMode.setEnabled(True)
            self.actions.createLineStripMode.setEnabled(True)
            self.actions.createAiPolygonMode.setEnabled(True)
            self._selectAiModelComboBox.setEnabled(False)
        else:
            if createMode == "polygon":
                self.actions.createMode.setEnabled(False)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
                self.actions.createAiPolygonMode.setEnabled(True)
                self._selectAiModelComboBox.setEnabled(False)
            elif createMode == "rectangle":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(False)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
                self.actions.createAiPolygonMode.setEnabled(True)
                self._selectAiModelComboBox.setEnabled(False)
            elif createMode == "line":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(False)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
                self.actions.createAiPolygonMode.setEnabled(True)
                self._selectAiModelComboBox.setEnabled(False)
            elif createMode == "point":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(False)
                self.actions.createLineStripMode.setEnabled(True)
                self.actions.createAiPolygonMode.setEnabled(True)
                self._selectAiModelComboBox.setEnabled(False)
            elif createMode == "circle":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(False)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
                self.actions.createAiPolygonMode.setEnabled(True)
                self._selectAiModelComboBox.setEnabled(False)
            elif createMode == "linestrip":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(False)
                self.actions.createAiPolygonMode.setEnabled(True)
                self._selectAiModelComboBox.setEnabled(False)
            elif createMode == "ai_polygon":
                self.actions.createMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
                self.actions.createAiPolygonMode.setEnabled(False)
                self.canvas.initializeAiModel(
                    name=self._selectAiModelComboBox.currentText()
                )
                self._selectAiModelComboBox.setEnabled(True)
            else:
                raise ValueError("Unsupported createMode: %s" % createMode)
        self.actions.editMode.setEnabled(not edit)

        

    def setEditMode(self):
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.newIcon("labels")
            action = QtWidgets.QAction(
                icon, "&%d %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(functools.partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def validateLabel(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.uniqLabelList.count()):
            label_i = self.uniqLabelList.item(i).data(Qt.UserRole)
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
                
        return False

    def editLabel(self, item=None):
        if item and not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem type")

        if not self.canvas.editing():
            return
        if not item:
            item = self.currentItem()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return
        text, flags, group_id, description = self.labelDialog.popUp(
            text=shape.label,
            flags=shape.flags,
            group_id=shape.group_id,
            description=shape.description,
        )
        if text is None:
            return
        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return
        shape.label = text
        shape.flags = flags
        shape.group_id = group_id
        shape.description = description

        self._update_shape_color(shape)
        if shape.group_id is None:
            item.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    html.escape(shape.label), *shape.fill_color.getRgb()[:3]
                )
            )
        else:
            item.setText("{} ({})".format(shape.label, shape.group_id))
        self.setDirty()
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)

        
    def change(self): # 체크박스 변경
            print("체크박스 변경")


    def fileSearchChanged(self):
        self.importDirImages(
            self.lastOpenDir,
            pattern=self.fileSearch.text(),
            load=False,
        )

    def fileSelectionChanged(self):
        items = self.fileListWidget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.mayContinue():
            return

        currIndex = self.imageList.index(str(item.text()))
        if currIndex < len(self.imageList):
            filename = self.imageList[currIndex]
            if filename:
                self.loadFile(filename)

    

    # React to canvas signals.
    def shapeSelectionChanged(self, selected_shapes):
        self._noSelectionSlot = True
        for shape in self.canvas.selectedShapes:
            shape.selected = False
        self.labelList.clearSelection()
        self.canvas.selectedShapes = selected_shapes
        for shape in self.canvas.selectedShapes:
            shape.selected = True
            item = self.labelList.findItemByShape(shape)
            self.labelList.selectItem(item)
            self.labelList.scrollToItem(item)
        self._noSelectionSlot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected == 1)

    def addLabel(self, shape):
        if shape.group_id is None:
            text = shape.label
        else:
            text = "{} ({})".format(shape.label, shape.group_id)
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        if self.uniqLabelList.findItemByLabel(shape.label) is None:
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        self.labelDialog.addLabelHistory(shape.label)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), *shape.fill_color.getRgb()[:3]
            )
        )

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label):
        if self._config["shape_color"] == "auto":
            item = self.uniqLabelList.findItemByLabel(label)
            if item is None:
                item = self.uniqLabelList.createItemFromLabel(label)
                self.uniqLabelList.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.uniqLabelList.setItemLabel(item, label, rgb)
            label_id = self.uniqLabelList.indexFromItem(item).row() + 1
            label_id += self._config["shift_auto_shape_color"]

            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        elif (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):

            return self._config["label_colors"][label]
        
        elif 'LPose' in label:
            return (0, 255, 0)
        elif 'RPose' in label:
            return (255, 255, 0)
        
        elif self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def remLabels(self, shapes):## 라벨 읽기 함수
        for shape in shapes:
            item = self.labelList.findItemByShape(shape)
            self.labelList.removeItem(item)

    def loadShapes(self, shapes, replace=True): ## 도형 읽기 함수
        self._noSelectionSlot = True
        for shape in shapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)

    def loadLabels(self, shapes):  ## 라벨 데이터 읽기 함수
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            description = shape.get("description", "")
            group_id = shape["group_id"]
            other_data = shape["other_data"]

            if not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                description=description,
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)
            shape.other_data = other_data

            s.append(shape)
        self.loadShapes(s)

    

    def saveLabels(self, filename):  ## 라벨데이터  저장함수 
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    description=s.description,
                    shape_type=s.shape_type,
                    flags=s.flags,
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        # for i in range(self.flag_widget.count()):
        #     item = self.flag_widget.item(i)
        #     key = item.text()
        #     flag = item.checkState() == Qt.Checked
        #     flags[key] = flag
        try:
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = self.imageData if self._config["store_data"] else None



            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=imagePath,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
            )
            self.labelFile = lf
            items = self.fileListWidget.findItems(
                self.imagePath, Qt.MatchExactly
            )
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename   
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False



## --------------------------------------------수정 X 기존 Labelme코드----------------------------------------------------------



    def duplicateSelectedShape(self):
        added_shapes = self.canvas.duplicateSelectedShapes()
        self.labelList.clearSelection()
        for shape in added_shapes:
            self.addLabel(shape)
        self.setDirty()

    def pasteSelectedShape(self):
        self.loadShapes(self._copied_shapes, replace=False)
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selectedShapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def labelSelectionChanged(self):
        if self._noSelectionSlot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.labelList.selectedItems():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.selectShapes(selected_shapes)
            else:
                self.canvas.deSelectShape()

    def labelItemChanged(self, item):
        shape = item.shape()
        self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelOrderChanged(self):
        self.setDirty()
        self.canvas.loadShapes([item.shape() for item in self.labelList])

    # Callback functions:

    def newShape(self):  ## 도형 그리기
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.uniqLabelList.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        description = ""

        if self._config["display_label_popup"] or not text:
            

            previous_text = self.labelDialog.edit.text()
            text, flags, group_id, description = self.labelDialog.popUp(text)
            if not text:
                self.labelDialog.edit.setText(previous_text)



        if text and not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""

        if text:
            print("이건머야 3")
            self.labelList.clearSelection()
            shape = self.canvas.setLastLabel(text, flags)
            shape.group_id = group_id
            shape.description = description
            self.addLabel(shape)
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
        else:
            self.canvas.undoLastLine()
            self.canvas.shapesBackups.pop()

        self.setEditMode()

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def setScroll(self, orientation, value):
        self.scrollBars[orientation].setValue(int(value))
        self.scroll_values[orientation][self.filename] = value

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def addZoom(self, increment=1.1):
        zoom_value = self.zoomWidget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.setZoom(zoom_value)

    def zoomRequest(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.setScroll(
                Qt.Horizontal,
                self.scrollBars[Qt.Horizontal].value() + x_shift,
            )
            self.setScroll(
                Qt.Vertical,
                self.scrollBars[Qt.Vertical].value() + y_shift,
            )

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def enableKeepPrevScale(self, enabled):
        self._config["keep_prev_scale"] = enabled
        self.actions.keepPrevScale.setChecked(enabled)

    def onNewBrightnessContrast(self, qimage):
        self.canvas.loadPixmap(
            QtGui.QPixmap.fromImage(qimage), clear_shapes=False
        )

    def brightnessContrast(self, value):
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        dialog.exec_()

        brightness = dialog.slider_brightness.value()
        contrast = dialog.slider_contrast.value()
        self.brightnessContrast_values[self.filename] = (brightness, contrast)

    def togglePolygons(self, value):
        for item in self.labelList:
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filename=None,OnlyLabel=False): ## ? 파일 불러오기... 
        """Load the specified file, or the last opened file if None."""
        # changing fileListWidget loads file

        if filename in self.imageList and (
            self.fileListWidget.currentRow() != self.imageList.index(filename)
        ):
            self.fileListWidget.setCurrentRow(self.imageList.index(filename))
            self.fileListWidget.repaint()
            return

        self.resetState()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)

        if not QtCore.QFile.exists(filename):
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False
        # assumes same name, but json extension
        self.status(
            str(self.tr("Loading %s...")) % osp.basename(str(filename))
        )
        label_file = osp.splitext(filename)[0] + ".json"
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
            label_file
        ):
            try:
                self.labelFile = LabelFile(label_file)
            except LabelFileError as e:
                self.errorMessage(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p>"
                        "<p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            self.imageData = self.labelFile.imageData
            self.imagePath = osp.join(
                osp.dirname(label_file),
                self.labelFile.imagePath,
            )
            self.otherData = self.labelFile.otherData
        else:
            self.imageData = LabelFile.load_image_file(filename)
            if self.imageData:
                self.imagePath = filename
            self.labelFile = None
        image = QtGui.QImage.fromData(self.imageData)

        if image.isNull():
            formats = [
                "*.{}".format(fmt.data().decode())
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.errorMessage(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.status(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.filename = filename
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        flags = {k: False for k in self._config["flags"] or []}
        if self.labelFile:
            self.loadLabels(self.labelFile.shapes)
            if self.labelFile.flags is not None:
                flags.update(self.labelFile.flags)
        #self.loadFlags(flags)
        if self._config["keep_prev"] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()
        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoomMode = self.zoom_values[self.filename][0]
            self.setZoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjustScale(initial=True)
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        # set brightness contrast values
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.imageData),
            self.onNewBrightnessContrast,
            parent=self,
        )
        brightness, contrast = self.brightnessContrast_values.get(
            self.filename, (None, None)
        )
        if self._config["keep_prev_brightness"] and self.recentFiles:
            brightness, _ = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if self._config["keep_prev_contrast"] and self.recentFiles:
            _, contrast = self.brightnessContrast_values.get(
                self.recentFiles[0], (None, None)
            )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        self.brightnessContrast_values[self.filename] = (brightness, contrast)
        if brightness is not None or contrast is not None:
            dialog.onNewValue(None)
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.canvas.setFocus()
        self.status(str(self.tr("Loaded %s")) % osp.basename(str(filename)))
        return True

    def resizeEvent(self, event):
        if (
            self.canvas
            and not self.image.isNull()
            and self.zoomMode != self.MANUAL_ZOOM
        ):
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        value = int(100 * value)
        self.zoomWidget.setValue(value)
        self.zoom_values[self.filename] = (self.zoomMode, value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def enableSaveImageWithData(self, enabled):
        self._config["store_data"] = enabled
        self.actions.saveWithImageData.setChecked(enabled)

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        self.settings.setValue(
            "filename", self.filename if self.filename else ""
        )
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("recentFiles", self.recentFiles)
        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    def dragEnterEvent(self, event): 
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any([i.lower().endswith(tuple(extensions)) for i in items]):
                event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.mayContinue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.importDroppedImageFiles(items)

    # User Dialogs #

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)



## --------------------------------------------수정 X 기존 Labelme코드----------------------------------------------------------




    def openPrevImg(self, _value=False):
        keep_prev = self._config["keep_prev"]
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        if self.filename is None:
            return

        currIndex = self.imageList.index(self.filename)
        if currIndex - 1 >= 0:
            filename = self.imageList[currIndex - 1]
            if filename:
                self.loadFile(filename)

        self._config["keep_prev"] = keep_prev


    def openNextImg(self, _value=False, load=True):
        keep_prev = self._config["keep_prev"]

        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if len(self.imageList) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.imageList[0]
        else:
            currIndex = self.imageList.index(self.filename)

            if currIndex + 1 < len(self.imageList):
                filename = self.imageList[currIndex + 1]

            else:## 끝도달

                if self.video_mode and currIndex<self.length-1:
                    print(str(self.Addcut) +"장씩 읽어옵니다")
                    self.vidcap=cv2.VideoCapture(self.vid_files[0])
                    count=0

                    self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, currIndex*self.fps) ## 현재 위치로

                    while(self.vidcap.isOpened()): ## 처음엔 장만
                        if(int(self.vidcap.get(1)) % self.fps == 0):
                            ret, image =  self.vidcap.read()

                            print(str(self.video_path)+str(currIndex+count))
                            self.imwriteKor(self.video_path+"_"+str(currIndex+count)+'.jpg',image,[cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                            
                            
                            
                            
                            count += 1
                            self.img_count+=1
                        else:
                            self.vidcap.grab()
                        

                        if(currIndex+count == self.length or count == self.Addcut): ## 10개씩 끊어서 ㅇㅇ
                            break

                    self.vidcap.release()


                    self.importDirImages(self.temp_name, load=False)



                    filename = self.imageList[-(count-1)]
                else:
                    filename=self.imageList[-1]

        self.filename = filename

        if self.filename and load:
            self.loadFile(self.filename)

        self._config["keep_prev"] = keep_prev


    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + ["*%s" % LabelFile.suffix]
        )
        fileDialog = FileDialogPreview(self)
        fileDialog.setFileMode(FileDialogPreview.ExistingFile)
        fileDialog.setNameFilter(filters)
        fileDialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        fileDialog.setWindowFilePath(path)
        fileDialog.setViewMode(FileDialogPreview.Detail)
        if fileDialog.exec_():
            fileName = fileDialog.selectedFiles()[0]
            if fileName:
                self.loadFile(fileName)

    def changeOutputDirDialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.currentPath()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()

        current_filename = self.filename
        self.importDirImages(self.lastOpenDir, load=False)

        if current_filename in self.imageList:
            # retain currently selected file
            self.fileListWidget.setCurrentRow(
                self.imageList.index(current_filename)
            )
            self.fileListWidget.repaint()

    def saveFile(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.labelFile:
            # DL20180323 - overwrite when in directory
            self._saveFile(self.labelFile.filename)
        elif self.output_file:
            self._saveFile(self.output_file)
            self.close()
        else:
            self._saveFile(self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.output_dir, filters
            )
        else:
            dlg = QtWidgets.QFileDialog(
                self, caption, self.currentPath(), filters
            )
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.currentPath(), basename + LabelFile.suffix
            )
        filename = dlg.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def getLabelFile(self):
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + ".json"

        return label_file

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, "
            "proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.getLabelFile()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info("Label file is removed: {}".format(label_file))

            item = self.fileListWidget.currentItem()
            item.setCheckState(Qt.Unchecked)

            self.resetState()

    # Message Dialogs. #
    def hasLabels(self):
        if self.noShapes():
            self.errorMessage(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def hasLabelFile(self):
        if self.filename is None:
            return False

        label_file = self.getLabelFile()
        return osp.exists(label_file)

    def mayContinue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr('Save annotations to "{}" before closing?').format(
            self.filename
        )
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        elif answer == mb.Save:
            self.saveFile()
            return True
        else:  # answer == mb.Cancel
            return False

    def errorMessage(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, "<p><b>%s</b></p>%s" % (title, message)
        )

    def currentPath(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggleKeepPrevMode(self):
        self._config["keep_prev"] = not self._config["keep_prev"]

    def removeSelectedPoint(self):
        self.canvas.removeSelectedPoint()
        self.canvas.update()
        if not self.canvas.hShape.points:
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape])
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)
        self.setDirty()

    def deleteSelectedShape(self):  ## 도형 삭제
        
        self.remLabels(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def copyShape(self): ## 도형 카피
        self.canvas.endMove(copy=True)
        for shape in self.canvas.selectedShapes:
            self.addLabel(shape)
        self.labelList.clearSelection()
        self.setDirty()

    def moveShape(self):## 도형이동
        self.canvas.endMove(copy=False)
        self.setDirty()

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else "."
        if self.lastOpenDir and osp.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = (
                osp.dirname(self.filename) if self.filename else "."
            )

        targetDirPath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                defaultOpenDirPath,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )
        print(targetDirPath)
        self.importDirImages(targetDirPath)

    @property
    def imageList(self):
        lst = []
        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            lst.append(item.text())
        return lst

    def importDroppedImageFiles(self, imageFiles):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        for file in imageFiles:
            if file in self.imageList or not file.lower().endswith(
                tuple(extensions)
            ):
                continue
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
                label_file
            ):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)


        if len(self.imageList) > 1:
            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)


        self.openNextImg()

    def addItemToWidget(self, dirpath, pattern=None):
        self.fileListWidget.clear()
        for filename in self.scanAllImages(dirpath):
            if pattern and pattern not in filename:
                continue
            label_file = osp.splitext(filename)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(self.output_dir, label_file_without_path)
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
                label_file
            ):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.fileListWidget.addItem(item)

    def importDirImages(self, dirpath, pattern=None, load=True):
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)
        self.actions.All_load.setEnabled(True) # 올로드
        self.actions.prev_img_bounding.setEnabled(True)# pre 바운딩
        #self.actions.Convert_to_Rectangle.setEnabled(True)
        #self.actions.Auto_Bounding.setEnabled(True)
        #self.actions.findUnlabeledFile.setEnabled(True)

        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.filename = None
        self.addItemToWidget(dirpath)
        self.openNextImg(load=load)

    def scanAllImages(self, folderPath):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        images = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(os.path.abspath(root), file)
                    images.append(relativePath)
        images = natsort.os_sorted(images)
        return images
    


### -----------------------------------------------------------------------------------------------------------수정사항

    def findUnlabeledFile(self):
        if not self.mayContinue():
            return
        
        if len(self.imageList) <= 0:
            return
        
        if self.filename is None:
            return

        last_checked_item = None

        for i in range(self.fileListWidget.count()):
            item = self.fileListWidget.item(i)
            if item.checkState() == Qt.Checked:
                last_checked_item= self.imageList[i]
        self.loadFile(last_checked_item)

    

    def AutoBounding_person(self): ### 오토 바인딩 -------------------------------------------------------------------------------------------
           
            #model.Find(self.lastOpenDir, self.lastOpenDir, cv2.imread(self.filename), os.path.basename(self.filename))
            if self.filename==None:
                return

            pose_detect.pose_detection(self.filename)

            self.loadFile(self.filename)# 다시오픈
            self.addItemToWidget(self.lastOpenDir)
            # print("구현중")# 사진을 받아와서 모델을 돌려 사각형 배치 할것!

    def AutoBounding_handBox(self): ### 오토 바인딩 -------------------------------------------------------------------------------------------
        
        if self.filename==None:
            return

        pose_detect.handBoxMake(self.filename)

        self.loadFile(self.filename)# 다시오픈
        self.addItemToWidget(self.lastOpenDir)

    def AutoBounding_PersonAndhand(self): ### 오토 바인딩 -------------------------------------------------------------------------------------------
        
        if self.filename==None:
            return

        pose_detect.PersonAndHand(self.filename)

        self.loadFile(self.filename)# 다시오픈
        self.addItemToWidget(self.lastOpenDir)


    def AutoBounding_hand(self): ### 오토 바인딩 -------------------------------------------------------------------------------------------
           
            #model.Find(self.lastOpenDir, self.lastOpenDir, cv2.imread(self.filename), os.path.basename(self.filename))
            if self.filename==None:
                return

            pose_detect.handDetect(self.filename)

            self.loadFile(self.filename)# 다시오픈
            
            print(self.filename)
            self.addItemToWidget(self.lastOpenDir)
            # print("구현중")# 사진을 받아와서 모델을 돌려 사각형 배치 할것!


    def All_autolabeling_person(self):
        print("전체 오토바인딩(사람)")
        print(self.filename+"위치에서부터 시작")

        if self.imageList==None:
            return
        
        nowIndex=self.imageList.index(self.filename)

        
        pose_detect.allfile_pose_detection(self.imageList,nowIndex)


        self.loadFile(self.imageList[nowIndex])
        self.addItemToWidget(self.lastOpenDir)


    def All_autolabeling_handBox(self):
            print("전체 오토바인딩(사람+박스)")
            print(self.filename+"위치에서부터 시작")

            if self.imageList==None:
                return
            
            nowIndex=self.imageList.index(self.filename)

            
            pose_detect.allfile_handBox_detection(self.imageList,nowIndex)


            self.loadFile(self.imageList[nowIndex])
            self.addItemToWidget(self.lastOpenDir)
    

    def All_autolabeling_PersonAndhand(self):
            print("전체 오토바인딩(사람)")
            print(self.filename+"위치에서부터 시작")

            if self.imageList==None:
                return
            
            nowIndex=self.imageList.index(self.filename)

            
            pose_detect.allfile_PersonHand_detection(self.imageList,nowIndex)


            self.loadFile(self.imageList[nowIndex])
            self.addItemToWidget(self.lastOpenDir)


    def All_autolabeling_hand(self):
        print("전체 오토바인딩(손)")
        print(self.filename+"위치에서부터 시작")

        if self.imageList==None:
            return
        

        nowIndex=self.imageList.index(self.filename)
        pose_detect.allfile_Hands_detection(self.imageList,nowIndex)


        self.loadFile(self.imageList[0])
        self.addItemToWidget(self.lastOpenDir)


    def AutoBounding_personSkel(self): ### 오토 바인딩 전신 관절 포인트 라벨링
           
            if self.filename==None:
                return

            pose_detect.Allpose_detection(self.filename)

            self.loadFile(self.filename)# 다시오픈
            self.addItemToWidget(self.lastOpenDir)

    
    def AutoBounding_personSkel_Body(self):  ##  관절 포인트 기반의 전신 관절 생성
        if self.filename==None:
            return

        pose_detect.PersonSkelMake(self.filename)

        self.loadFile(self.filename)# 다시오픈
        self.addItemToWidget(self.lastOpenDir)


    def All_AutoBounding_personSkel(self): ### 오토 바인딩 전신 관절 포인트 라벨링 전체

        if self.imageList==None:
            return
        

        nowIndex=self.imageList.index(self.filename)
        pose_detect.allfile_Allpose_detection(self.imageList,nowIndex)


        self.loadFile(self.imageList[0])
        self.addItemToWidget(self.lastOpenDir)


    def All_AutoBounding_personSkel_Body(self): ##  관절 포인트 기반의 전신 관절 생성 전체 

        if self.imageList==None:
            return
        

        nowIndex=self.imageList.index(self.filename)
        pose_detect.allfile_PersonSkelMake(self.imageList,nowIndex)


        self.loadFile(self.imageList[0])
        self.addItemToWidget(self.lastOpenDir)


    def AutoBounding_personSkel_remove(self):  ##  관절 포인트 기반의 전신 관절 삭제

        if self.filename==None:
            return

        pose_detect.AllposeReset(self.filename)

        self.loadFile(self.filename)# 다시오픈
        self.addItemToWidget(self.lastOpenDir)


    def handChange(self):
        print("손 방향 반대로")
        

        if self.selectPersonCombo=="":
            return
        
        if self.filename==None:
            return

        pose_detect.handChange(self.selectPersonCombo,self.filename)

        self.loadFile(self.filename)# 다시오픈
        self.addItemToWidget(self.lastOpenDir)



    def OnePerson(self):
        print("재탐색")
        

        if self.selectPersonCombo=="":
            return
        
        if self.filename==None:
            return

        pose_detect.ReDetectPerson(self.selectPersonCombo,self.filename)


        self.loadFile(self.filename)# 다시오픈
        self.addItemToWidget(self.lastOpenDir)



    def PreLabel_F(self): #앞선 json손 불러오기
        print("앞선손")

        if self.filename==None:
            return


        PersonNum=self.precombo1.currentText()
        HandNum=self.precombo2.currentText()

        currIndex = self.imageList.index(self.filename)

        if currIndex>0:  ## 앞에 있는지 확인
            prev_file_path = self.imageList[currIndex - 1]
            prev_file_path=os.path.splitext(prev_file_path)[0]+".json"

            file_name, ext = os.path.splitext(self.filename)
            file_path = file_name + '.json'

            pose_detect.PreHandLabel(prev_file_path,file_path,PersonNum,HandNum)


            self.loadFile(self.filename)# 다시오픈
            self.addItemToWidget(self.lastOpenDir)
        






    def ActionLabeling(self): ## 1. 손 두개를 묶는   2. 별개로   >> json구성
        print("액션라벨링 시작")
        ActionResult=[]

        if self.filename==None:
            return

        for i in range(len(self.ActionComboList)):
            selected_text = self.ActionComboList[i].currentText()
            ActionResult.append(selected_text)



        pose_detect.PersonAction(self.filename,ActionResult)            


        # Display the selected text in a QLabel
        #self.result_label.setText(f"Selected Value: {selected_text}")


        self.loadFile(self.filename)# 다시오픈
        self.addItemToWidget(self.lastOpenDir)


## --------------------------------------  비디오 로드 -----------------------------------------------------------------------------
    def Load_Video_save(self): ## 비디오 사진을 저장하는부분
        files, ext = QtWidgets.QFileDialog.getOpenFileNames(self
                                                , 'Select one or more files to open'
                                                , ''
                                                , 'Video (*.mp4 *.mpg *.mpeg *.avi *.wma *.mkv *mov))')
        
        if files:
            self.video_mode=True
            self.vid_files=files
            self.vidcap=cv2.VideoCapture(self.vid_files[0])
            self.temp_name=os.path.dirname(files[0]) +"/"+os.path.basename(files[0][:-4])+"_"+str(self.fps)+"fps_Js"
            self.default_save_dir = self.temp_name

            self.video_path = os.path.abspath(self.temp_name)

            self.temp_name=self.video_path


            self.video_path=os.path.join(self.video_path,os.path.basename(files[0][:-4]))## 비디오 이름 을 추가하는 부분입니다  // 제외해도 무관
            self.vid_count=0


            self.length = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT)/self.fps)


            


            if(not os.path.exists(self.temp_name)): ## 파일 존재 x
                print("파일이 존재 하지 않아 새로 생성합니다.")
                os.mkdir(self.temp_name)
            else: ## 파일이 존재 할시
                self.importDirImages(self.temp_name)

                #self.actions.Auto_Bounding.setEnabled(True)
                #self.actions.findUnlabeledFile.setEnabled(True)
                self.actions.copy.setEnabled(True)
                self.actions.prev_img_bounding.setEnabled(True)
                
                self.vidcap.release()
                
                return

            countFive=0
            while(self.vidcap.isOpened()): ## 처음엔 5장만
                if(int(self.vidcap.get(1)) % self.fps == 0):
                    ret, image =  self.vidcap.read()

                    #print(str(self.video_path)+str(countFive))

                    file_path = self.video_path+"_"+str(countFive)+'.jpg'
                    #print(file_path)

                    self.imwriteKor(file_path,image,[cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    
                    
                    countFive += 1
                    self.img_count+=1
                else:
                    self.vidcap.grab()
                
                ## self.img_count  >> 현재 까지 읽어들인 비디오 사진수 저장
                ## countFive >> 처음엔 5장만
                if(countFive == self.length or countFive == 5): ## 5개씩 끊어서 ㅇㅇ
                    break

            self.vidcap.release()
            self.importDirImages(self.temp_name)
            
            

            #self.actions.Auto_Bounding.setEnabled(True)
            #self.actions.findUnlabeledFile.setEnabled(True)
            self.actions.copy.setEnabled(True)
            self.actions.prev_img_bounding.setEnabled(True)

    #------------------------------------------------------ALL 로드---------------------------------------------------------

        
    def video_move(self):
        print(len(self.video_list))

    

    def All_load(self): ## 로드
        ind_count=0
        
        
        currIndex = self.imageList.index(self.filename)
        self.img_count = len(self.imageList)

        if self.length<=self.img_count:   ## >> 현재위치에서 끝까지 이동
            print("현재위치에서 All 로드 시작합니다")
            currIndex+=1

            while(currIndex<self.img_count):
                print("실행중 . . . "+str(currIndex) +" / " + str(self.img_count))
                currIndex+=1
                
                self.openNextImg()

            print("ALL Load 끝!")
            return
        
            

        elif self.video_mode:## 한번에 로드하는 부분 (비디오일 경우)
            print("여기서는 사진만 읽어 저장 >> 마지막엔 디렉토리 읽기")


            self.vidcap=cv2.VideoCapture(self.vid_files[0])
            stand=len(self.imageList)
            add_count=1

            self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, stand*self.fps) ## 이동

            while(self.vidcap.isOpened()):

                # if(stand+add_count == self.length or add_count>=self.Addcut): ## Addcut 만큼만 다시저장
                #     break
                if(stand+ind_count==self.length):
                    break

                if(int(self.vidcap.get(1))% self.fps==0): ## ffmpeg로 변경해도 괜찮을거 같은데?
                    ret, image =  self.vidcap.read()

                    print("추가됨 : " + str(stand+ind_count+1) +" / "+str(self.length) + "    주소 : "+str(self.video_path+"_"+str(stand+ind_count)+'.jpg'))
                    
                    self.imwriteKor(self.video_path+"_"+str(stand+ind_count)+'.jpg',image,[cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    
                    ind_count+=1# 현재 인덱스?
                    add_count+=1
                else:
                    self.vidcap.grab()
                    

            self.importDirImages(self.temp_name)
            self.img_count = len(self.imageList)

            self.vidcap.release()

    #-----------------------------------------------prev 바운딩-------------------------------------------------------------------


    def prev_img_bounding(self):  ## 앞선 json 불러오기
        currIndex = self.imageList.index(self.filename)

        if currIndex>0:  ## 앞에 있는지 확인
            prev_file_path = self.imageList[currIndex - 1]

            prev_file_path=os.path.splitext(prev_file_path)[0]+".json"

            if(os.path.exists(prev_file_path)):
                file_path=os.path.splitext(self.imageList[currIndex])[0]+".json"

                if(not os.path.exists(file_path)): #현재 파일 없다면 생성
                    self.saveLabels(file_path) ## 현재 라벨 저장


                with open(file_path, "r") as file:## 값 추출
                    json_data = json.load(file)

                image_path = json_data["imagePath"]
                os.remove(file_path)


                shutil.copyfile(prev_file_path, file_path)  ## 복사하기

                with open(file_path, "r") as file:# 수정
                    json_data = json.load(file)

                # 값을 변경
                json_data["imagePath"] = image_path

                with open(file_path, "w") as file:# 저장
                    json.dump(json_data, file, indent=4)

                self.loadFile(self.filename)# 다시오픈
    


    def PolygonToRectangle(self):  ## 파일 단위로 변경?
        print("다각형을 사각형으로 변경합니다.")

        for i in range(0,len(self.imageList)):
            currIndex = i
            

            currJson = os.path.splitext(self.imageList[currIndex])[0]+".json"

            if not os.path.exists(currJson):
                print("파일없음")
                return

            print(currJson)


            with open(currJson, 'r') as file:
                    json_data = json.load(file)  # JSON 파일 읽기

            x=[]
            y=[]

            for shape in json_data['shapes']:
                if shape['shape_type'] !="rectangle":
                    for point in shape['points']:
                        x.append(point[0])
                        y.append(point[1])
                        

                        x_min = min(x)
                        x_max = max(x)
                        y_min = min(y)
                        y_max = max(y)

                    shape['shape_type']="rectangle"
                    shape['points']=[[x_min,y_min],[x_max,y_max]]



            with open(currJson, 'w') as file:
                    json.dump(json_data, file, indent=4) 


        print("모두 변경했습니다.")
