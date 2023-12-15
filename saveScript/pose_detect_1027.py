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


Person_Save_List={}## 현재 이미지에서의 사람 박스 바인딩 정보 // /대기




Person_Save_Hand_List={} ## 이전 손 인식 라벨링 정보들

Person_List={}## 현재 이미지에서의 사람 박스 라벨링 정보


def dist(point1,point2):
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
        

def pose_detection(image_path):
    model = YOLO('yolov8x-pose.pt')
    file_name, ext = os.path.splitext(image_path)
    
    jsonfile_name = file_name + '.json'

    image = cv2.imread(image_path)


    results = model.predict(image, save=False,half=True, iou=0.5, conf=0.3)  ## ------------------------- 욜로 모델 설정


    height, width, channels = image.shape


    if os.path.isfile(jsonfile_name):
        with open(jsonfile_name, 'r') as file:
            input_data = file.read()
            label_json = json.loads(input_data)
    else:
        print(os.path.basename(image_path))
        
        
        label_json = {
            "version": "0.0.0",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }
    
    table_pos = [
        (720, 850),  # 1
        (600, 341),  # 2
        (1450, 60),  # 3
        (1740, 1900),  # dealer
        (2450, 70),  # 5
        (3140, 496),  # 6
        (3420, 1150)  # 7
    ]



    
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) ## 좌상단 우하단 좌표로 이루어진 좌표
    confs = results[0].boxes.conf.cpu().numpy().astype(float)
    keypoints = results[0].keypoints.xy.cpu().numpy().astype(int)



    # label_json = {
    #     "version": "0.0.0",
    #     "flags": {},
    #     "shapes": [],
    #     "imagePath": image_path,
    #     "imageData": None,
    #     "imageHeight": height,
    #     "imageWidth": width
    # }

    
    
    Person_List.clear()

    for i in range(len(boxes)):  ## 플레이어들
        x1, y1, x2, y2 = boxes[i]
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



        ## 박스를 그리기전!

        if p_name not in Person_List:

            Person_List[p_name]=[]


        Person_List[p_name].append([min_x, min_y,max_x, max_y,i])


    for Person in Person_List:  ## 플레이어 박스 그리기  // /가장 큰걸로
        
        if Person_List[Person]==[]:
            continue

        Temp_IND=PersonSizeCheck(Person_List[Person])
        keypoint = keypoints[Person_List[Person][Temp_IND][4]]
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


        handDetect(image,xyposition,keypoint,label_json,Person,0)




    

    # DealerTable=[980,800,2750,2150]  ### 딜러 구역-------------------------------------


    # Tempimg = image[DealerTable[1]:DealerTable[3], DealerTable[0]:DealerTable[2]]
    

    # results = model.predict(Tempimg, save=False)



    # boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) ## 좌상단 우하단 좌표로 이루어진 좌표들
    # confs = results[0].boxes.conf.cpu().numpy().astype(float)
    # keypoints = results[0].keypoints.xy.cpu().numpy().astype(int)


    
    # bestPreDealer_IND=0
    # temp_y=0
    # size=0
    # temp=0.5


    # for i in range(len(boxes)):## 제일 하단에 있는 하나만 하기   ## 수정 필요
    #     x1, y1, x2, y2 = boxes[i]


    #     # 넓이가 가장 큰거로  ㄱㄱㄱㄱ

        

    #     #temp_miny=min(y1,y2) ## y축 으로 
    #     # if temp_y>temp_miny:
    #     #     temp_y=temp_miny
    #     #     bestPreDealer_IND=i
        
        
    #     temp_size=(x2-x1)*(y2-y1)## 큰 사각형으로 선택
    #     if temp_size>size:
    #         size=temp_size
    #         bestPreDealer_IND=i

    


    # if len(boxes)>bestPreDealer_IND:
    #     #for i in range(len(boxes)): ## 딜러만
    #     x1, y1, x2, y2 = boxes[bestPreDealer_IND]
    #     min_x, min_y = min(x1, x2), min(y1, y2)
    #     max_x, max_y = max(x1, x2), max(y1, y2)

    #     p_name = 'Dealer'
    #     p_prefix = "D"



    #     label_json['shapes'].append({ ### ----------------------------수정해야됨
    #         "label": p_name,
    #         "points": [
    #             [DealerTable[0]+min_x, DealerTable[1]+min_y], 
    #             [DealerTable[0]+max_x, DealerTable[1]+max_y]
    #         ],
    #         "group_id": None,
    #         "description": "",
    #         "shape_type": "rectangle",
    #         "flags": {}
    #     })


    #     keypoint = keypoints[bestPreDealer_IND]

    #     for point in keypoint:
    #         point[0]+=DealerTable[0]
    #         point[1]+=DealerTable[1]
            


    #     xyposition=[DealerTable[0]+min_x, DealerTable[1]+min_y, DealerTable[0]+max_x, DealerTable[1]+max_y]

    #     handDetect(image,xyposition,keypoint,label_json,p_prefix,1)## 딜러시작
        








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


def handDetect(image,xyposition,keypoint,label_json,p_prefix, Dealer):


    handBoxSize=150  ## 각각의 손 바인딩 박스 크기


    basePoint_R=keypoint[10]  ## 손목 부분 체크
    basePoint_L=keypoint[9]

    height, width, channels = image.shape
     

    label_json['shapes'].append({
        "label": f'{p_prefix}_Skeleton',
        "points": [
            keypoint[9],
            keypoint[7],
            keypoint[5],
            keypoint[6],
            keypoint[8],
            keypoint[10]
        ],
        "group_id": None,
        "description": "",
        "shape_type": "linestrip",
        "flags": {}
    })



    

    ## 손의 가운데 지점 좌표
    DetectPoint_R,DetectPoint_L=RHHand_Line_extens(keypoint[10],keypoint[8],keypoint[9],keypoint[7])



    
    R_Box=[[int(DetectPoint_R[0]-handBoxSize),int(DetectPoint_R[1]-handBoxSize)],[int(DetectPoint_R[0]+handBoxSize),int(DetectPoint_R[1]+handBoxSize)]]
    L_Box=[[int(DetectPoint_L[0]-handBoxSize),int(DetectPoint_L[1]-handBoxSize)],[int(DetectPoint_L[0]+handBoxSize),int(DetectPoint_L[1]+handBoxSize)]]
    

    DectectHandBox=[R_Box,L_Box]


    ## 현재 손 박스 2개를 합쳐서 인식중
    
    #------------------------------- 손이 있어야할 자리
    # label_json['shapes'].append({ 
    #             "label": 'testBox_R',
    #             "points": [
    #                 R_Box[0],
    #                 R_Box[1]
    #             ],
    #             "group_id": None,
    #             "description": "",
    #             "shape_type": "rectangle",
    #             "flags": {}
    #         })
    

    # label_json['shapes'].append({ 
    #             "label": 'testBox_L',
    #             "points": [
    #                 L_Box[0],
    #                 L_Box[1]
    #             ],
    #             "group_id": None,
    #             "description": "",
    #             "shape_type": "rectangle",
    #             "flags": {}
    #         })






    
    ## ------------------------- 손 부분만 따로 인식 돌리는 역할 필
    # 이미지를 어느정도 짜르기


    # if Dealer:
    #     min_x=xyposition[0]
    #     min_y=xyposition[1]

    #     max_x=max(keypoint[7][0],keypoint[8][0])
    #     max_x=max(max_x,xyposition[2])

    #     max_y=max(keypoint[7][1],keypoint[8][1])
        
        

    # else:
    #     min_x=min(keypoint[7][0],keypoint[8][0])
    #     min_x=min(min_x,xyposition[0])

    #     min_y=min(keypoint[7][1],keypoint[8][1])

    #     max_x=xyposition[2]
    #     max_y=xyposition[3]
        

    
  
    min_x=max(0,min(DectectHandBox[0][0][0],DectectHandBox[1][0][0]))
    min_y=max(0,min(DectectHandBox[0][0][1],DectectHandBox[1][0][1]))
    max_x=min(width,max(DectectHandBox[0][1][0],DectectHandBox[1][1][0]))
    max_y=min(height,max(DectectHandBox[0][1][1],DectectHandBox[1][1][1]))


    label_json['shapes'].append({ 
            "label": 'testBox_RRRRRRRRRRRRR',
            "points": [
                [min_x,min_y],
                [max_x,max_y]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {}
        })



    # label_json['shapes'].append({ 
    #             "label": 'testBox',
    #             "points": [
    #                 [min_x, min_y], 
    #                 [max_x, max_y]
    #             ],
    #             "group_id": None,
    #             "description": "",
    #             "shape_type": "rectangle",
    #             "flags": {}
    #         })
    



    
    # hands detecting per person

    Hands_List=[]
    Hands_Box_List=[]
    Hands_Log_List=[]
    Hands_Label_List=[]
    
    checkbool=0
    hand=""

    finger =['Thum','Index','Middle','Ring','Pinky']
    
    def dectect(point): ## 손 인식 부분
        

        img = image[point[1]:point[3], point[0]:point[2]]  # [min_y:max_y, min_x:max_x]

        
        checkbool=0
        hand=""

        #cv2.imwrite('test'+str(point[1])+'.jpg', img)

        # print("-------------------------------------------------")
        # print(p_prefix)
        # print(str(point[0])+ ","+ str(point[1])+","+str(point[2])+ ","+ str(point[3]))

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

    
    




    ## json 손 스켈레톤
    def RLHand_json(RLstr,IND):

        label_json['shapes'].append({
            "label": f'{p_prefix}_'+RLstr+'H',
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
                "label": f'{p_prefix}_'+RLstr+'H_'+f'{finger[i]}'+'_skel',
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
            



            
    if(checkbool==1):##손 하나만

        if str(hand)=='Left':

            handBox_minuSize=50  ## 박스 축소
            min_x=max(0,L_Box[0][0]+handBox_minuSize)
            min_y=max(0,L_Box[0][1]+handBox_minuSize)
            max_x=min(width,L_Box[1][0]-handBox_minuSize)
            max_y=min(height,L_Box[1][1]-handBox_minuSize)


            label_json['shapes'].append({ 
                "label": 'testBox_L',
                "points": [
                    [min_x,min_y],
                    [max_x,max_y]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            })

           

            RLHand_json('R',0)

            Hands_List=[]
            Hands_Box_List=[]
            Hands_Log_List=[]
            Hands_Label_List=[]

            hand,checkbool=dectect([min_x,min_y,max_x,max_y])
            #RH_IND,LH_IND=RHCheck(basePoint_R,basePoint_L,Hands_List)


            for i in range(len(Hands_Label_List)):
                if Hands_Label_List[i][0]=='L':
                    RLHand_json('L',i)

        else:

            handBox_minuSize=50  ## 박스 축소
            min_x=max(0,R_Box[0][0]+handBox_minuSize)
            min_y=max(0,R_Box[0][1]+handBox_minuSize)
            max_x=min(width,R_Box[1][0]-handBox_minuSize)
            max_y=min(height,R_Box[1][1]-handBox_minuSize)



            label_json['shapes'].append({ 
                "label": 'testBox_R',
                "points": [
                    [min_x,min_y],
                    [max_x,max_y]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            })


            
            RLHand_json('L',0)
            Hands_List=[]
            Hands_Box_List=[]
            Hands_Log_List=[]
            Hands_Label_List=[]

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
            

        
        
        
        

            

        


def allfile_pose_detection(imageList):

    for file in imageList:
        pose_detection(image_path=file)


if __name__ == "__main__":
    pose_detection('./pose_test/[004][Top4K]20230524_180000(20230616153801)_194.jpg')
