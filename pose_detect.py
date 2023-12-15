import os
import cv2
import json
import math

import numpy as np
import mediapipe as mp

from typing import Any
from ultralytics import YOLO
 

class JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super(JSONEncoder, self).default(o)

os.environ['KMP_DUPLICATE_LIB_OK']='True'



## 미디어 파이프 설정
mp_hands = mp.solutions.hands
_hands = mp_hands.Hands(static_image_mode=True,
      max_num_hands=10,
      min_detection_confidence=0.05) ## 4개중 base손목과 가까운거 2개 선정
model = YOLO('yolov8x-pose.pt')



Person_Save_List={}## 현재 이미지에서의 사람 박스 바인딩 정보 // /대기

Person_Save_Hand_List={} ## 이전 손 인식 라벨링 정보들

Person_List={}## 현재 이미지에서의 사람 박스 라벨링 정보



table_pos = [
        (720, 850),  # 1
        (600, 341),  # 2
        (1450, 60),  # 3
        (1740, 1900),  # dealer
        (2450, 70),  # 5
        (3140, 496),  # 6
        (3420, 1150)  # 7
    ]


def dist(point1,point2): # 단순 거리 계산
    return math.sqrt((point1[0]-point2[0])*(point1[0]-point2[0])+(point1[1]-point2[1])*(point1[1]-point2[1]))


def RHCheck(basePoint_R,basePoint_L,hands): ## 손목지점에서 각 박스가 가까운 지점찾기
    RH=None
    LH=None
    
    for i in range(len(hands)):
        if RH==None or dist(hands[i][0],basePoint_R)<dist(hands[RH][0],basePoint_R):
            RH=i
        
        if LH==None or dist(hands[i][0],basePoint_L)<dist(hands[LH][0],basePoint_L):
            LH=i

    return RH,LH



def RHHand_Line_extens(point10,point8,point9,point7): ## 오른쪽, 왼쪽 손 박스 관련
    
    DetectPoint_R=None
    DetectPoint_L=None

     ## 손의 좌표
    def cal(point):
        
        dx = point[0][0] - point[1][0]
        dy = point[0][1] - point[1][1]

        length = math.sqrt(dx**2 + dy**2)

        if length==0:
            return point[0]


        unit_vector_x = dx / length
        unit_vector_y = dy / length
        
        Ax = point[0][0] + 100 * unit_vector_x
        Ay = point[0][1] + 100 * unit_vector_y

        return [Ax,Ay]
    

    DetectPoint_R=cal([point10,point8])
    DetectPoint_L=cal([point9,point7])



    return DetectPoint_R,DetectPoint_L
    

def PersonSizeCheck(boxList):  ### 가장 큰 사각형 반환
    if len(boxList)==1:
        return 0
    
    size=0
    bestPreDealer_IND=0

    for i in range(len(boxList)):
        temp_size=(boxList[i][2]-boxList[i][0])*(boxList[i][3]-boxList[i][1])## 큰 사각형으로 선택

        if temp_size>size:
            size=temp_size
            bestPreDealer_IND=i


    return bestPreDealer_IND
   


def PersonSkelPoint(boxes,keypoints,label_json,width,height):  ## 플레이어 박스 및 관절

    for i in range(len(boxes)):  ## 플레이어들
        x1, y1, x2, y2 = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        min_x, min_y = min(x1, x2), min(y1, y2)
        max_x, max_y = max(x1, x2), max(y1, y2)
        cen_x, cen_y = (x1 + x2) // 2, (y1 + y2) // 2

        tpos_arr = np.array(table_pos)
        tpos_arr -= (cen_x, cen_y)
        dist_t = np.sqrt(np.sum(tpos_arr**2, axis=1))
        
        p_num = np.argmin(dist_t)
        
        p_name ="Person"
        p_prefix="P"

        p_name='P1'

        ## 박스를 그리기전!

        if p_name not in Person_List:
            Person_List[p_name]=[]


        Person_List[p_name].append([min_x, min_y,max_x, max_y,i])


    for Person in Person_List:  ## 플레이어 박스 그리기  // /가장 큰걸로
        
        if Person_List[Person]==[]:
            continue

        Temp_IND=PersonSizeCheck(Person_List[Person])
        
        xyposition=[Person_List[Person][Temp_IND][0],Person_List[Person][Temp_IND][1],Person_List[Person][Temp_IND][2],Person_List[Person][Temp_IND][3]]

        Person="P1"
        
        label_json['shapes'].append({
            "label": Person,  # +str(confs[i])
            "points": [
                [xyposition[0], xyposition[1]], 
                [xyposition[2], xyposition[3]]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {}
        })


        
        keypoint=keypoints[Person_List[Person][Temp_IND][4]]

        label_json['shapes'].append({
            "label": "S1",
            "points": [
                keypoint[8]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "point",
            "flags": {}
            })
        
        label_json['shapes'].append({
            "label": "S2",
            "points": [
                keypoint[6]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "point",
            "flags": {}
            })

        label_json['shapes'].append({
            "label": "S3",
            "points": [
                keypoint[5]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "point",
            "flags": {}
            })
        
        label_json['shapes'].append({
            "label": "S4",
            "points": [
                keypoint[7],
            ],
            "group_id": None,
            "description": "",
            "shape_type": "point",
            "flags": {}
            })
        

        label_json['shapes'].append({
            "label": "S5",
            "points": [
                keypoint[14],
            ],
            "group_id": None,
            "description": "",
            "shape_type": "point",
            "flags": {}
            })
        

        label_json['shapes'].append({
            "label": "S6",
            "points": [
                keypoint[12],
            ],
            "group_id": None,
            "description": "",
            "shape_type": "point",
            "flags": {}
            })
        
        
        label_json['shapes'].append({
            "label": "S7",
            "points": [
                keypoint[11],
            ],
            "group_id": None,
            "description": "",
            "shape_type": "point",
            "flags": {}
            })
        

        label_json['shapes'].append({
            "label": "S8",
            "points": [
                keypoint[13],
            ],
            "group_id": None,
            "description": "",
            "shape_type": "point",
            "flags": {}
            })
        

def PersonSkelMake(image_path): ## 관절포인트 기반 스켈레톤 구성
        file_name, ext = os.path.splitext(image_path)
    
        jsonfile_name = file_name + '.json'

        if os.path.isfile(jsonfile_name):
            with open(jsonfile_name, 'r') as file:
                print("---------------")
                print(jsonfile_name)

                input_data = file.read()
                label_json = json.loads(input_data)
        else:
            print(os.path.basename(image_path))


        keypoints = [None] * 9

        for shape in label_json['shapes']:
            label = shape['label']
            if label.startswith('S') and label[1:]:
                index = int(label[1:])  # 'P1'에서 숫자를 추출하고 0부터 시작하는 인덱스로 변환
                points = shape['points']
                if points:
                    keypoints[index] = points[0]




        label_json['shapes'].append({
        "label": 'P_Skeleton_1',
        "points": [
            #keypoint[9],#0
            keypoints[1],
            keypoints[2],
            keypoints[3],
            keypoints[4],
            #keypoint[10]#5
        ],
        "group_id": None,
        "description": "",
        "shape_type": "linestrip",
        "flags": {}
        })


        label_json['shapes'].append({
            "label": 'P_Skeleton_2',
            "points": [
                keypoints[5],
                keypoints[6],
                keypoints[2],
                keypoints[3],
                keypoints[7],
                keypoints[8]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "linestrip",
            "flags": {}
        })


        label_json['shapes'].append({
            "label": 'P_Skeleton_3',
            "points": [
                keypoints[6],
                keypoints[7]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "line",
            "flags": {}
        })


        j_text = json.dumps(label_json, cls=JSONEncoder)
        with open(f'{file_name}.json', 'w') as j:
            j.write(j_text)

        image=None



def PersonBoxMake(boxes,keypoints,label_json,width,height):  ## 플레이어 박스 및 상반신 관절 

    for i in range(len(boxes)):  ## 플레이어들
        x1, y1, x2, y2 = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        min_x, min_y = min(x1, x2), min(y1, y2)
        max_x, max_y = max(x1, x2), max(y1, y2)
        cen_x, cen_y = (x1 + x2) // 2, (y1 + y2) // 2

        tpos_arr = np.array(table_pos)
        tpos_arr -= (cen_x, cen_y)
        dist_t = np.sqrt(np.sum(tpos_arr**2, axis=1))
        
        p_num = np.argmin(dist_t)
        

        p_name ="Person"
        p_prefix="P"

        if p_num == 3:  ## 탑뷰일때만
            p_name = 'D'
            p_prefix = "D"

            # continue  ## 딜러는 따로 실행할것
        else:
            p_name = f'P{p_num+1}'
            p_prefix = f'P{p_num+1}'

        if p_name=='P7':  ## 플레이어 7번 빼는 부분---------------------
            continue

        ## 박스를 그리기전!

        if p_name not in Person_List:
            Person_List[p_name]=[]


        Person_List[p_name].append([min_x, min_y,max_x, max_y,i])


    for Person in Person_List:  ## 플레이어 박스 그리기  // /가장 큰걸로
        
        if Person_List[Person]==[]:
            continue

        Temp_IND=PersonSizeCheck(Person_List[Person])
        
        xyposition=[Person_List[Person][Temp_IND][0],Person_List[Person][Temp_IND][1],Person_List[Person][Temp_IND][2],Person_List[Person][Temp_IND][3]]


        label_json['shapes'].append({
            "label": Person,  # +str(confs[i])
            "points": [
                [xyposition[0], xyposition[1]], 
                [xyposition[2], xyposition[3]]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {}
        })

        ## 재인식
        # tempimage=image[xyposition[1]:xyposition[3], xyposition[0]:xyposition[2]]  # [min_y:max_y, min_x:max_x]

        # result_person = model.predict(tempimage, save=False,half=True, iou=0.5, conf=0.5)
        
        #keypoint = result_person[0].keypoints.xy.cpu().numpy().astype(int)
        keypoint=keypoints[Person_List[Person][Temp_IND][4]]


        label_json['shapes'].append({
        "label": Person+'_Skeleton',
        "points": [
            keypoint[9],#0
            keypoint[7],
            keypoint[5],
            keypoint[6],
            keypoint[8],
            keypoint[10]#5
        ],
        "group_id": None,
        "description": "",
        "shape_type": "linestrip",
        "flags": {}
        })

       

def handBoxMake(image_path): ## 손 박스만 만들기 (손목의 위치에 따라)
        
        file_name, ext = os.path.splitext(image_path)
    
        jsonfile_name = file_name + '.json'

        image = cv2.imread(image_path)
        height, width, channels = image.shape


        if os.path.isfile(jsonfile_name):
            with open(jsonfile_name, 'r') as file:
                input_data = file.read()
                label_json = json.loads(input_data)
        else:
            print(os.path.basename(image_path))


        handBoxSize=150  ## 각각의 손 바인딩 박스 크기

        Person_List=["P1","P2","P3","P5","P6","P7","D"]


        for person in Person_List:## 관절 찾기

            target_key  = person+"_Skeleton"
            keypoint=None

            handBoxSize=150  ## 각각의 손 바인딩 박스 크기

            for shape in label_json["shapes"]:
                if shape["label"]==target_key:## 사람 찾아서  키포인트 반환
                    keypoint=shape["points"]

                    DetectPoint_R,DetectPoint_L=RHHand_Line_extens(keypoint[5],keypoint[4],keypoint[0],keypoint[1])
                        
                    R_Box=[[int(DetectPoint_R[0]-handBoxSize),int(DetectPoint_R[1]-handBoxSize)],[int(DetectPoint_R[0]+handBoxSize),int(DetectPoint_R[1]+handBoxSize)]]
                    L_Box=[[int(DetectPoint_L[0]-handBoxSize),int(DetectPoint_L[1]-handBoxSize)],[int(DetectPoint_L[0]+handBoxSize),int(DetectPoint_L[1]+handBoxSize)]]
                        

                    handBox_minuSize=10  ## 박스 축소
                    min_x=max(0,L_Box[0][0]+handBox_minuSize)
                    min_y=max(0,L_Box[0][1]+handBox_minuSize)
                    max_x=min(width,L_Box[1][0]-handBox_minuSize)
                    max_y=min(height,L_Box[1][1]-handBox_minuSize)


                    new_shapes = [shape for shape in label_json["shapes"] if shape["label"] != person+'_LBox' and shape["label"] != person+'_RBox']
                    label_json["shapes"] = new_shapes




                    label_json['shapes'].append({ ## 각각의 손에 박스 바운딩
                        "label": person+'_LBox',
                        "points": [
                            [min_x,min_y],
                            [max_x,max_y]
                        ],
                        "group_id": None,
                        "description": "",
                        "shape_type": "rectangle",
                        "flags": {}
                    })


                    min_x=max(0,R_Box[0][0]+handBox_minuSize)
                    min_y=max(0,R_Box[0][1]+handBox_minuSize)
                    max_x=min(width,R_Box[1][0]-handBox_minuSize)
                    max_y=min(height,R_Box[1][1]-handBox_minuSize)

                    label_json['shapes'].append({ 
                        "label": person+'_RBox',
                        "points": [
                            [min_x,min_y],
                            [max_x,max_y]
                        ],
                        "group_id": None,
                        "description": "",
                        "shape_type": "rectangle",
                        "flags": {}
                    })


        j_text = json.dumps(label_json, cls=JSONEncoder)
        with open(f'{file_name}.json', 'w') as j:
            j.write(j_text)

        image=None


def Allpose_detection(image_path):  ## 전신 관절 포인트 측정
    
    file_name, ext = os.path.splitext(image_path)
    
    jsonfile_name = file_name + '.json'

    image = cv2.imread(image_path)
    height, width, channels = image.shape

    results = model.predict(image, save=False,device=0,half=True, iou=0.5, conf=0.3)  ## ------------------------- 욜로 모델 설정

    if os.path.isfile(jsonfile_name):
        with open(jsonfile_name, 'r') as file:
            input_data = file.read()
            label_json = json.loads(input_data)
    else:
        print(os.path.basename(image_path))
        
        
    label_json = {## 탐색할때마다 초기화 시킬까 말까
        "version": "0.0.0",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }


    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) ## 좌상단 우하단 좌표로 이루어진 좌표
    confs = results[0].boxes.conf.cpu().numpy().astype(float)
    keypoints = results[0].keypoints.xy.cpu().numpy().astype(int)

    Person_List.clear()

    PersonSkelPoint(boxes=boxes,keypoints=keypoints,label_json=label_json,width=width,height=height)

    j_text = json.dumps(label_json, cls=JSONEncoder)
    with open(f'{file_name}.json', 'w') as j:
        j.write(j_text)


    image=None


def AllposeReset(image_path): ## 기존 전신 관절 라벨링 삭제

    file_name, ext = os.path.splitext(image_path)
    
    jsonfile_name = file_name + '.json'

    if os.path.isfile(jsonfile_name):
        with open(jsonfile_name, 'r') as file:
            input_data = file.read()
            label_json = json.loads(input_data)
    else:
        print(os.path.basename(image_path))


    label_to_remove=["P_Skeleton_1","P_Skeleton_2","P_Skeleton_3"]

    for removeObj in label_to_remove:
        new_shapes = [shape for shape in label_json["shapes"] if shape["label"] != removeObj]
        label_json["shapes"] = new_shapes


    j_text = json.dumps(label_json, cls=JSONEncoder)
    with open(f'{file_name}.json', 'w') as j:
        j.write(j_text)
    

def pose_detection(image_path):  ## yolo 시작  # 상반신 탑뷰에서의 상반신 관절포인트 측정
    
    file_name, ext = os.path.splitext(image_path)
    
    jsonfile_name = file_name + '.json'

    image = cv2.imread(image_path)
    height, width, channels = image.shape

    results = model.predict(image, save=False,device=0,half=True, iou=0.5, conf=0.3)  ## ------------------------- 욜로 모델 설정

    if os.path.isfile(jsonfile_name):
        with open(jsonfile_name, 'r') as file:
            input_data = file.read()
            label_json = json.loads(input_data)
    else:
        print(os.path.basename(image_path))
        
        
    label_json = {## 탐색할때마다 초기화 시킬까 말까
        "version": "0.0.0",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }


    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) ## 좌상단 우하단 좌표로 이루어진 좌표
    confs = results[0].boxes.conf.cpu().numpy().astype(float)
    keypoints = results[0].keypoints.xy.cpu().numpy().astype(int)

    Person_List.clear()

    
        
    PersonBoxMake(boxes=boxes,keypoints=keypoints,label_json=label_json,width=width,height=height)

        ## 손 인식 잠시 중단
        ##handDetect(image,xyposition,keypoint,label_json,Person)# keypoint[0]



#--------------------------------
            
    ## 사람json추가 --------------------------------------------------


    # for points in keypoints:
    #     cv2.circle(image, points[9], 3, (255, 0, 255), 5)
    #     cv2.circle(image, points[10], 3, (255, 0, 255), 5)

    # cv2.imwrite(f'{file_name}.jpg', image)
    # cv2.waitKey(0)

    j_text = json.dumps(label_json, cls=JSONEncoder)
    with open(f'{file_name}.json', 'w') as j:
        j.write(j_text)


    image=None


def handDetect(image_path):##  손 인식 + 그리기
    file_name, ext = os.path.splitext(image_path)
    
    jsonfile_name = file_name + '.json'

    image = cv2.imread(image_path)
    height, width, channels = image.shape


    Person_List=["P1","P2","P3","P5","P6","P7","D"]

    if os.path.exists(jsonfile_name):## 사람과 관절 수정이 끝났다면 그 결과로 실행
        with open(jsonfile_name, 'r') as json_file:

            input_data = json_file.read()
            label_json = json.loads(input_data)

        for person in Person_List:## 관절 찾기
            target_key_L =person+"_LBox"
            target_key_R =person+"_RBox"
            target_key  =person+"_Skeleton"
            
            value=None
            value_L=None ## 관절 포인트
            value_R=None ## 관절 포인트
            
            handBoxSize=150  ## 각각의 손 바인딩 박스 크기

            for shape in label_json["shapes"]:
                if shape["label"] == target_key_L:
                    value_L = shape

                elif shape["label"] == target_key_R:
                    value_R = shape

                elif shape["label"]==target_key:
                    value=shape
            
            
            if value is not None:
                keypoint=value["points"]

                basePoint_R=keypoint[5]
                basePoint_L=keypoint[0]

                # 손박스 생성및 관리
                if value_R is None or value_L is None:
                    DetectPoint_R,DetectPoint_L=RHHand_Line_extens(keypoint[5],keypoint[4],keypoint[0],keypoint[1])
            
                    R_Box=[[int(DetectPoint_R[0]-handBoxSize),int(DetectPoint_R[1]-handBoxSize)],[int(DetectPoint_R[0]+handBoxSize),int(DetectPoint_R[1]+handBoxSize)]]
                    L_Box=[[int(DetectPoint_L[0]-handBoxSize),int(DetectPoint_L[1]-handBoxSize)],[int(DetectPoint_L[0]+handBoxSize),int(DetectPoint_L[1]+handBoxSize)]]

                    if value_L is None:
                        handBox_minuSize=10  ## 박스 축소
                        min_x=max(0,L_Box[0][0]+handBox_minuSize)
                        min_y=max(0,L_Box[0][1]+handBox_minuSize)
                        max_x=min(width,L_Box[1][0]-handBox_minuSize)
                        max_y=min(height,L_Box[1][1]-handBox_minuSize)


                        label_json['shapes'].append({ ## 각각의 손에 박스 바운딩
                            "label": person+'_LBox',
                            "points": [
                                [min_x,min_y],
                                [max_x,max_y]
                            ],
                            "group_id": None,
                            "description": "",
                            "shape_type": "rectangle",
                            "flags": {}
                        })

                    if value_R is None:
                        handBox_minuSize=10  ## 박스 축소
                        min_x=max(0,R_Box[0][0]+handBox_minuSize)
                        min_y=max(0,R_Box[0][1]+handBox_minuSize)
                        max_x=min(width,R_Box[1][0]-handBox_minuSize)
                        max_y=min(height,R_Box[1][1]-handBox_minuSize)


                        label_json['shapes'].append({ ## 각각의 손에 박스 바운딩
                            "label": person+'_RBox',
                            "points": [
                                [min_x,min_y],
                                [max_x,max_y]
                            ],
                            "group_id": None,
                            "description": "",
                            "shape_type": "rectangle",
                            "flags": {}
                        })
                else:
                    R_Box=value_R["points"]
                    L_Box=value_L["points"]
                
            else:
                print(person+"관절없음 사람을 지우거나, 재인식 필요")
                continue


            height, width, channels = image.shape
            

            ## 손의 가운데 지점 좌표
            


            DectectHandBox=[R_Box,L_Box]

        
            min_x=max(0,min(DectectHandBox[0][0][0],DectectHandBox[1][0][0]))
            min_y=max(0,min(DectectHandBox[0][0][1],DectectHandBox[1][0][1]))
            max_x=min(width,max(DectectHandBox[0][1][0],DectectHandBox[1][1][0]))
            max_y=min(height,max(DectectHandBox[0][1][1],DectectHandBox[1][1][1]))


            # label_json['shapes'].append({ 
            #         "label": person+'_Hands',
            #         "points": [
            #             [min_x,min_y],
            #             [max_x,max_y]
            #         ],
            #         "group_id": None,
            #         "description": "",
            #         "shape_type": "rectangle",
            #         "flags": {}
            #     })
            

            Hands_List=[]
            Hands_Box_List=[]
            Hands_Log_List=[]
            Hands_Label_List=[]
            
            checkbool=0
            hand=""

            finger =['Thum','Index','Middle','Ring','Pinky']
            
            def dectect(point): ## 손 인식 부분
                img = image[int(point[1]):int(point[3]), int(point[0]):int(point[2])]  # [min_y:max_y, min_x:max_x]

                checkbool=0
                hand=""

                results = _hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                wrists = []

                # print(results.multi_handedness[0].classification[0].label)

                if results.multi_hand_landmarks:
                    # print(results.multi_handedness[0])
                    handedness = results.multi_handedness
                    hand_landmarks = results.multi_hand_landmarks
                    
                    if(len(hand_landmarks)==1):## 한쪽손만 인식됨
                        checkbool=1

                    elif(len(hand_landmarks)>=2):
                        checkbool=2


                    for i in range(len(hand_landmarks)):
                        all_x = []
                        all_y = []
                        all_point=[]

                        for hnd in mp_hands.HandLandmark:
                            all_x.append(hand_landmarks[i].landmark[hnd].x)
                            all_y.append(hand_landmarks[i].landmark[hnd].y)
                            
                            all_point.append([hand_landmarks[i].landmark[hnd].x* img.shape[1]+point[0]
                                            ,hand_landmarks[i].landmark[hnd].y* img.shape[0]+point[1],hnd])

                        hand_x1 = int(np.min(all_x) * img.shape[1] + point[0])
                        hand_y1 = int(np.min(all_y) * img.shape[0] + point[1])
                        hand_x2 = int(np.max(all_x) * img.shape[1] + point[0])
                        hand_y2 = int(np.max(all_y) * img.shape[0] + point[1])

                        hand_x1 = 0 if hand_x1 - 10 < 0 else hand_x1 - 10
                        hand_y1 = 0 if hand_y1 - 10 < 0 else hand_y1 - 10
                        hand_x2 = width - 1 if hand_x2 + 10 >= width else hand_x2 + 10
                        hand_y2 = height - 1 if hand_y2 + 10 >= height else hand_y2 + 10

                        wrist_x, wrist_y = hand_landmarks[i].landmark[mp_hands.HandLandmark.WRIST].x, hand_landmarks[i].landmark[mp_hands.HandLandmark.WRIST].y


                        x, y = int(wrist_x * img.shape[1] + point[0]), int(wrist_y * img.shape[0] + point[1])
                        wrists.append([x, y])


                        # print(handedness[i].classification[0].label)
                        hand = handedness[i].classification[0].label

                        
                        Hands_List.append(all_point)
                        Hands_Box_List.append([hand_x1,hand_y1,hand_x2,hand_y2])
                        Hands_Log_List.append(handedness[i].classification[0].score)
                        Hands_Label_List.append(hand)

                else:
                    print("인식된 정보 없음!")
                    
                return hand,checkbool

            hand,checkbool=dectect([min_x,min_y,max_x,max_y])
            

            def Hands_reset(RLstr,IND): ## 기존 라벨링 삭제 
                label_to_remove=f'{person}_'+RLstr+'H'

                new_shapes = [shape for shape in label_json["shapes"] if shape["label"] != label_to_remove]
                label_json["shapes"] = new_shapes

                for i in range(0,5):

                    label_to_remove=f'{person}_'+RLstr+'H_'+f'{finger[i]}'+'_skel'
                
                    new_shapes = [shape for shape in label_json["shapes"] if shape["label"] != label_to_remove]
                    label_json["shapes"] = new_shapes

            ## json 손 스켈레톤
            def RLHand_json(RLstr,IND):

                Hands_reset(RLstr,IND)

                label_json['shapes'].append({
                    "label": f'{person}_'+RLstr+'H',
                    "points": [
                        (Hands_Box_List[IND][0], Hands_Box_List[IND][1]), 
                        (Hands_Box_List[IND][2], Hands_Box_List[IND][3])
                    ],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {}
                })

                for i in range(0,5):
                    label_json['shapes'].append({
                        "label": f'{person}_'+RLstr+'H_'+f'{finger[i]}'+'_skel',
                        "points": [
                            #[basePoint_R[0],basePoint_R[1]],
                            [Hands_List[IND][i*4+1][0],Hands_List[IND][i*4+1][1]],
                            [Hands_List[IND][i*4+2][0],Hands_List[IND][i*4+2][1]],
                            [Hands_List[IND][i*4+3][0],Hands_List[IND][i*4+3][1]],
                            [Hands_List[IND][i*4+4][0],Hands_List[IND][i*4+4][1]]
                        ],
                        "group_id": None,
                        "description": "",
                        "shape_type": "linestrip",
                        "flags": {}
                })


            if(checkbool==0):

                Hands_List=[]
                Hands_Box_List=[]
                Hands_Log_List=[]
                Hands_Label_List=[]

                min_x=max(0,L_Box[0][0])
                min_y=max(0,L_Box[0][1])
                max_x=min(width,L_Box[1][0])
                max_y=min(height,L_Box[1][1])

                

                hand,checkbool=dectect([min_x,min_y,max_x,max_y])

                for i in range(len(Hands_Label_List)):
                        if Hands_Label_List[i][0]=='L':
                            RLHand_json('L',i)
                            break


                Hands_List=[]
                Hands_Box_List=[]
                Hands_Log_List=[]
                Hands_Label_List=[]

                min_x=max(0,R_Box[0][0])
                min_y=max(0,R_Box[0][1])
                max_x=min(width,R_Box[1][0])
                max_y=min(height,R_Box[1][1])

                

                hand,checkbool=dectect([min_x,min_y,max_x,max_y])

                for i in range(len(Hands_Label_List)):
                    if Hands_Label_List[i][0]=='R':
                        RLHand_json('R',i)
                        break

            elif(checkbool==1):##손 하나만
                if str(hand)=='Left':

                    RLHand_json('R',0)

                    Hands_List=[]
                    Hands_Box_List=[]
                    Hands_Log_List=[]
                    Hands_Label_List=[]

                    min_x=max(0,L_Box[0][0])
                    min_y=max(0,L_Box[0][1])
                    max_x=min(width,L_Box[1][0])
                    max_y=min(height,L_Box[1][1])

                    hand,checkbool=dectect([min_x,min_y,max_x,max_y])
                    #RH_IND,LH_IND=RHCheck(basePoint_R,basePoint_L,Hands_List)

                    for i in range(len(Hands_Label_List)):
                        if Hands_Label_List[i][0]=='L':
                            RLHand_json('L',i)

                else:

                    RLHand_json('L',0)

                    Hands_List=[]
                    Hands_Box_List=[]
                    Hands_Log_List=[]
                    Hands_Label_List=[]
                    
                    min_x=max(0,R_Box[0][0])
                    min_y=max(0,R_Box[0][1])
                    max_x=min(width,R_Box[1][0])
                    max_y=min(height,R_Box[1][1])

                    hand,checkbool=dectect([min_x,min_y,max_x,max_y])
                    #RH_IND,LH_IND=RHCheck(basePoint_R,basePoint_L,Hands_List)
                    for i in range(len(Hands_Label_List)):
                        if Hands_Label_List[i][0]=='R':
                            RLHand_json('R',i)


                ## 손 하나더 추적하기
                
            elif checkbool==2:

                RH_IND,LH_IND=RHCheck(basePoint_R,basePoint_L,Hands_List)  ## 제일 정확한거 인덱스로 넣어주기    >>> 지금은 제일 가까운거....

                RLHand_json('R',RH_IND)
                RLHand_json('L',LH_IND)
        

        j_text = json.dumps(label_json, cls=JSONEncoder)
        with open(f'{file_name}.json', 'w') as j:
            j.write(j_text)
        image=None

    else:
        pose_detection(image_path)## 결과가 없었으면 yolo 돌리고 실행

        handDetect(image_path)




    # for j in range(len(Hands_Box_List)):
    #     label_json['shapes'].append({
    #         "label": f'{p_prefix}_alllllll',
    #         "points": [
    #             (Hands_Box_List[j][0], Hands_Box_List[j][1]), 
    #             (Hands_Box_List[j][2], Hands_Box_List[j][3])
    #         ],
    #         "group_id": None,
    #         "description": "",
    #         "shape_type": "rectangle",
    #         "flags": {}
    #     })




    
    # for j in range(len(Hands_Box_List)):
    #     label_json['shapes'].append({
    #     "label": "testHand"+str(Hands_Log_List[j]),
    #     "points": [
    #         (Hands_Box_List[j][0], Hands_Box_List[j][1]), 
    #         (Hands_Box_List[j][2], Hands_Box_List[j][3])
    #     ],
    #     "group_id": None,
    #     "description": "",
    #     "shape_type": "rectangle",
    #     "flags": {}
    # })
            

def ReDetectPerson(person,image_path):  ## 사람 재인식 
    file_name, ext = os.path.splitext(image_path)
    
    jsonfile_name = file_name + '.json'

    image = cv2.imread(image_path)
    height, width, channels = image.shape


    if os.path.isfile(jsonfile_name):
        with open(jsonfile_name, 'r') as file:
            print("---------------")
            print(jsonfile_name)

            input_data = file.read()
            label_json = json.loads(input_data)
    else:
        print(os.path.basename(image_path))


    def json_reset(str):
        label_to_remove=str

        new_shapes = [shape for shape in label_json["shapes"] if shape["label"] != label_to_remove]
        label_json["shapes"] = new_shapes


    Personpoint=[]
    for shape in label_json["shapes"]:
        label_to_Person=person
        
        if shape["label"] == label_to_Person:
            Personpoint=shape["points"]
            continue

    
    
    if Personpoint==[]:
        return


    plusSize=50
    Px1=max(0,Personpoint[0][0]-plusSize)
    Py1=max(0,Personpoint[0][1]-plusSize)
    Px2=min(width,Personpoint[1][0]+plusSize)
    Py2=min(height,Personpoint[1][1]+plusSize)


    tempimage=image[int(Py1):int(Py2), int(Px1):int(Px2)]  # [min_y:max_y, min_x:max_x]
    Reresult = model.predict(tempimage, save=False,device=0,half=True, iou=0.5, conf=0.3)  ## ------------------------- 욜로 모델 설정


    boxes = Reresult[0].boxes.xyxy.cpu().numpy().astype(int) ## 좌상단 우하단 좌표로 이루어진 좌표
    confs = Reresult[0].boxes.conf.cpu().numpy().astype(float)
    keypoints = Reresult[0].keypoints.xy.cpu().numpy().astype(int)

    if len(boxes)==0:
        print("인식결과 없음")

        return

    boxList=[]

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i][0]+Px1,boxes[i][1]+Py1,boxes[i][2]+Px1,boxes[i][3]+Py1
        min_x, min_y = min(x1, x2), min(y1, y2)
        max_x, max_y = max(x1, x2), max(y1, y2)
        boxList.append([min_x, min_y,max_x, max_y,i])

    if len(boxList)==0:
        return

    Temp_IND=PersonSizeCheck(boxList)


    xyposition=[boxList[Temp_IND][0],boxList[Temp_IND][1],boxList[Temp_IND][2],boxList[Temp_IND][3]]


    json_reset(person)

    label_json['shapes'].append({
        "label": person,  # +str(confs[i])
        "points": [
            [xyposition[0], xyposition[1]], 
            [xyposition[2], xyposition[3]]
        ],
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {}
    })


    keypoint=keypoints[boxList[Temp_IND][4]]


    for point in keypoint:## 시작지점 보정 
        point[0]+=Px1
        point[1]+=Py1
        

    json_reset(person+'_Skeleton')
    label_json['shapes'].append({
    "label": person+'_Skeleton',
    "points": [
        keypoint[9],#0
        keypoint[7],
        keypoint[5],
        keypoint[6],
        keypoint[8],
        keypoint[10]#5
    ],
    "group_id": None,
    "description": "",
    "shape_type": "linestrip",
    "flags": {}
})
    
    handBoxSize=150  ## 각각의 손 바인딩 박스 크기


    DetectPoint_R,DetectPoint_L=RHHand_Line_extens(keypoint[10],keypoint[8],keypoint[9],keypoint[7])
        
    R_Box=[[int(DetectPoint_R[0]-handBoxSize),int(DetectPoint_R[1]-handBoxSize)],[int(DetectPoint_R[0]+handBoxSize),int(DetectPoint_R[1]+handBoxSize)]]
    L_Box=[[int(DetectPoint_L[0]-handBoxSize),int(DetectPoint_L[1]-handBoxSize)],[int(DetectPoint_L[0]+handBoxSize),int(DetectPoint_L[1]+handBoxSize)]]
        


    handBox_minuSize=10  ## 박스 축소
    min_x=max(0,L_Box[0][0]+handBox_minuSize)
    min_y=max(0,L_Box[0][1]+handBox_minuSize)
    max_x=min(width,L_Box[1][0]-handBox_minuSize)
    max_y=min(height,L_Box[1][1]-handBox_minuSize)


    json_reset(person+'_LBox')
    label_json['shapes'].append({ ## 각각의 손에 박스 바운딩
        "label": person+'_LBox',
        "points": [
            [min_x,min_y],
            [max_x,max_y]
        ],
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {}
    })


    min_x=max(0,R_Box[0][0]+handBox_minuSize)
    min_y=max(0,R_Box[0][1]+handBox_minuSize)
    max_x=min(width,R_Box[1][0]-handBox_minuSize)
    max_y=min(height,R_Box[1][1]-handBox_minuSize)


    json_reset(person+'_RBox')
    label_json['shapes'].append({ 
        "label": person+'_RBox',
        "points": [
            [min_x,min_y],
            [max_x,max_y]
        ],
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {}
    })



    j_text = json.dumps(label_json, cls=JSONEncoder)
    with open(f'{file_name}.json', 'w') as j:
        j.write(j_text)


    image=None


def handChange(person,image_path):  ## 손 방향 변경

    file_name, ext = os.path.splitext(image_path)
    
    jsonfile_name = file_name + '.json'



    find_RList=["_RH","_RH_Thum_skel","_RH_Index_skel","_RH_Middle_skel","_RH_Ring_skel","_RH_Pinky_skel"]
    find_LList=["_LH","_LH_Thum_skel","_LH_Index_skel","_LH_Middle_skel","_LH_Ring_skel","_LH_Pinky_skel"]
    
    print("--------- 시작")

    if os.path.exists(jsonfile_name):## 사람과 관절 수정이 끝났다면 그 결과로 실행
        with open(jsonfile_name, 'r') as json_file:

            input_data = json_file.read()
            label_json = json.loads(input_data)

        
            


        for shape in label_json["shapes"]:

            for i in range(len(find_RList)):
                label_to_remove_R=person+find_RList[i]
                label_to_remove_L=person+find_LList[i]
                
                if shape["label"] == label_to_remove_L:
                    shape["label"] = label_to_remove_R
                    continue

                if shape["label"] == label_to_remove_R:
                    shape["label"] = label_to_remove_L
                    continue

        

        j_text = json.dumps(label_json, cls=JSONEncoder)
        with open(f'{file_name}.json', 'w') as j:
            j.write(j_text)


def PersonAndHand(image_path): ## 사람과 손박스 동시에
    pose_detection(image_path=image_path)
    handBoxMake(image_path=image_path)


def PreHandLabel(prepath,path,person,Hand):# 앞선 손 json데이터 불러오기

    if Hand=='RL':
        PreHandLabel(prepath,path,person,'RH')
        PreHandLabel(prepath,path,person,'LH')
        return

    HandLabel=person+"_"+Hand

    labelList=["","_Thum_skel","_Index_skel","_Middle_skel",
               "_Pinky_skel","_Ring_skel"]
    

    file_name, ext = os.path.splitext(path)

    
    with open(path, 'r') as file:
        json_data = json.load(file)
        for label in labelList:
            Delete_label(HandLabel+label,json_data)


    with open(prepath, 'r') as file:
        json_data_pre = json.load(file)

        for label in labelList:
            saveLabel=find_label(HandLabel+label,json_data_pre)
            if saveLabel!=None:
                json_data["shapes"].append(saveLabel)
    
        
    j_text = json.dumps(json_data, cls=JSONEncoder)
    with open(f'{file_name}.json', 'w') as j:
        j.write(j_text)


def Delete_label(name,json_data): ## 기존 라벨링 삭제 
    label_to_remove=name

    new_shapes = [shape for shape in json_data["shapes"] if shape["label"] != label_to_remove]
    json_data["shapes"] = new_shapes


def find_label(name,json_data):## 라벨 찾아 반환하기
    for shape in json_data["shapes"]:
            if "label" in shape and shape["label"] == name:
                return shape
            
    return None


def PersonAction(image_path,actionList):## 액션라벨링
    print("액션 라벨링 시작")
    personList=["P1","P2","P3","P5","P6","P7","D"]

    file_name, ext = os.path.splitext(image_path)
    
    jsonfile_name = file_name + '.json'

    image = cv2.imread(image_path)
    height, width, channels = image.shape


    if os.path.isfile(jsonfile_name):
        with open(jsonfile_name, 'r') as file:
            input_data = file.read()
            label_json = json.loads(input_data)
    else:
        print(os.path.basename(image_path))


    index=0

    for person in personList:## 액션 라벨링 초기화


        label_to_remove=[person+'_RPose',person+'_LPose',person+'_RLPose',person+'_LBox',person+'_RBox']

        new_shapes = [shape for shape in label_json["shapes"] if not any(label in shape["label"] for label in label_to_remove)]

        label_json["shapes"] = new_shapes

        # 손 받아오기


        Rhand=None
        Lhand=None
        
    
        for shape in label_json["shapes"]:
            if "label" in shape:
                if shape["label"] == person+'_RH':
                    Rhand=shape["points"]
                if shape["label"] == person+'_LH':
                    Lhand=shape["points"]
                
        
        # -------- 두손인지 한손인지 판별하는 부분필요  >> 거리 순
        #- ------------------------------

        handsDist=500 ## 초기화
        size=30


        # if Rhand!=None and Lhand!=None:
            
        #     handsDist=dist([(Rhand[0][0]+Rhand[1][0])/2,(Rhand[0][1]+Rhand[1][1])/2],
        #          [(Lhand[0][0]+Lhand[1][0])/2,(Lhand[0][1]+Lhand[1][1])/2])

        # actionList[index]
        # actionList[index+1]

        


        if handsDist<=150:## 두손이 붙어있다?
            minx=min(Rhand[0][0],Lhand[0][0])
            miny=min(Rhand[0][1],Lhand[0][1])
            maxx=max(Rhand[1][0],Lhand[1][0])
            maxy=max(Rhand[1][1],Lhand[1][1])
            

            label_json['shapes'].append({
                            "label": person+'_RLPose_'+actionList[index],
                            "points": [
                                [max(minx-size,0),max(miny-size,0)],
                                [min(maxx+size,width),min(maxy+size,height)]
                            ],
                            "group_id": None,
                            "description": "",
                            "shape_type": "rectangle",
                            "flags": {}
                        })

        else:
            if Rhand!=None:
                label_json['shapes'].append({ ## 각각의 손에 박스 바운딩
                                "label": person+'_RPose_'+actionList[index],
                                "points": [
                                    [max(Rhand[0][0]-size,0),max(Rhand[0][1]-size,0)],
                                    [min(Rhand[1][0]+size,width),min(Rhand[1][1]+size,height)]
                                ],
                                "group_id": None,
                                "description": "",
                                "shape_type": "rectangle",
                                "flags": {}
                            })
            

            if Lhand!=None:
                label_json['shapes'].append({ ## 각각의 손에 박스 바운딩
                                "label": person+'_LPose_'+actionList[index+1],
                                "points": [
                                    [max(Lhand[0][0]-size,0),max(Lhand[0][1]-size,0)],
                                    [min(Lhand[1][0]+size,width),min(Lhand[1][1]+size,height)]
                                ],
                                "group_id": None,
                                "description": "",
                                "shape_type": "rectangle",
                                "flags": {}
                            })


        index+=2

        


    j_text = json.dumps(label_json, cls=JSONEncoder)
    with open(f'{file_name}.json', 'w') as j:
        j.write(j_text)

        
    image=None
    

## 이미지 리스트 전체 디텍션
def allfile_pose_detection(imageList,index):## 이미지 리스트 전체 상반신 관절 디텍션
    for i in range(index,len(imageList)):
        pose_detection(image_path=imageList[i])

    
def allfile_handBox_detection(imageList,index): ## 이미지 리스트 전체 핸드박스 그리기
    for i in range(index,len(imageList)):
        handBoxMake(image_path=imageList[i])


def allfile_Hands_detection(imageList,index): ## 이미지 리스트 전체 사람 + 손 박스 그리기
    for i in range(index,len(imageList)):
        handDetect(image_path=imageList[i])


def allfile_PersonHand_detection(imageList,index): ## 이미지 리스트 전체 손 관절 그리기
    for i in range(index,len(imageList)):
        PersonAndHand(image_path=imageList[i])


def allfile_Allpose_detection(imageList,index): ## 이미지 리스트 전체 관절 포인트 그리기
    for i in range(index,len(imageList)):
        Allpose_detection(image_path=imageList[i])


def allfile_PersonSkelMake(imageList,index): # 이미지 리스트 전체 관절 그리기
    for i in range(index,len(imageList)):
        PersonSkelMake(image_path=imageList[i])


