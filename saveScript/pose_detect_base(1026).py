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



mp_hands = mp.solutions.hands
_hands = mp_hands.Hands(static_image_mode=False,min_detection_confidence=0.01,max_num_hands=4) ## 4개중 base손목과 가까운거 2개 선정



def dist(point1,point2):
    return math.sqrt((point1[0]-point2[0])*(point1[0]-point2[0])+(point1[1]-point2[1])*(point1[1]-point2[1]))

def RHCheck(basePoint_R,basePoint_L,hands):
    RH=None
    LH=None
    
    for i in range(len(hands)):
        if RH==None or dist(hands[i][0],basePoint_R)<dist(hands[RH][0],basePoint_R):
            RH=i
        
        if LH==None or dist(hands[i][0],basePoint_L)<dist(hands[LH][0],basePoint_L):
            LH=i

    return RH,LH


def pose_detection(image_path):
    model = YOLO('yolov8x-pose.pt')
    file_name, ext = os.path.splitext(image_path)
    
    jsonfile_name = file_name + '.json'



    image = cv2.imread(image_path)
    results = model.predict(image, save=False)

    height, width, channels = image.shape


    if os.path.isfile(jsonfile_name):
        with open(jsonfile_name, 'r') as file:
            input_data = file.read()
            label_json = json.loads(input_data)
    else:
        print(os.path.basename(image_path))
        print(image_path)
        
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

    

    for i in range(len(boxes)):  ## 플레이어들
        if confs[i] > 0.5:
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
                p_name = 'Dealer'
                p_prefix = "D"

                continue  ## 딜러는 따로 실행할것
            else:
                p_name = f'P{p_num+1}'
                p_prefix = f'P{p_num+1}'


            label_json['shapes'].append({
                "label": p_name,
                "points": [
                    [min_x, min_y], 
                    [max_x, max_y]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            })



            keypoint = keypoints[i]
            xyposition=[min_x,min_y,max_x,max_y]
            
                    
            handDetect(image,xyposition,keypoint,label_json,p_prefix,0)




    

    DealerTable=[980,800,2750,2150]  ### 딜러 구역-------------------------------------


    Tempimg = image[DealerTable[1]:DealerTable[3], DealerTable[0]:DealerTable[2]]

    results = model.predict(Tempimg, save=False)



    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) ## 좌상단 우하단 좌표로 이루어진 좌표들
    confs = results[0].boxes.conf.cpu().numpy().astype(float)
    keypoints = results[0].keypoints.xy.cpu().numpy().astype(int)


    
    bestPreDealer_IND=0
    temp_y=0
    temp=0.5



    for i in range(len(boxes)):## 제일 하단에 있는 하나만 하기 
        x1, y1, x2, y2 = boxes[i]

        temp_maxy=max(y1,y2)
        if temp_y<temp_maxy:
            temp_y=temp_maxy
            bestPreDealer_IND=i
        
            

    


    if len(boxes)>bestPreDealer_IND:
        #for i in range(len(boxes)): ## 딜러만
        x1, y1, x2, y2 = boxes[bestPreDealer_IND]
        min_x, min_y = min(x1, x2), min(y1, y2)
        max_x, max_y = max(x1, x2), max(y1, y2)

        p_name = 'Dealer'
        p_prefix = "D"



        label_json['shapes'].append({ ### ----------------------------수정해야됨
            "label": p_name,
            "points": [
                [DealerTable[0]+min_x, DealerTable[1]+min_y], 
                [DealerTable[0]+max_x, DealerTable[1]+max_y]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {}
        })


        keypoint = keypoints[bestPreDealer_IND]

        for point in keypoint:
            point[0]+=DealerTable[0]
            point[1]+=DealerTable[1]
            


        xyposition=[DealerTable[0]+min_x, DealerTable[1]+min_y, DealerTable[0]+max_x, DealerTable[1]+max_y]

        handDetect(image,xyposition,keypoint,label_json,p_prefix,1)
        

            
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

    


    basePoint_R=keypoint[9]  ## 손목 부분 체크
    basePoint_L=keypoint[10]

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

    
    ## ------------------------- 손 부분만 따로 인식 돌리는 역할 필
    # 이미지를 어느정도 짜르기


    if Dealer:
        min_x=xyposition[0]
        min_y=xyposition[1]

        max_x=max(keypoint[7][0],keypoint[8][0])
        max_x=max(max_x,xyposition[2])

        max_y=max(keypoint[7][1],keypoint[8][1])
        
        

    else:
        min_x=min(keypoint[7][0],keypoint[8][0])
        min_x=min(min_x,xyposition[0])

        min_y=min(keypoint[7][1],keypoint[8][1])

        max_x=xyposition[2]
        max_y=xyposition[3]
        
        
    

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
    img = image[min_y:max_y, min_x:max_x]  # [min_y:max_y, min_x:max_x]



    results = _hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    wrists = []

    # print(results.multi_handedness[0].classification[0].label)



    if results.multi_hand_landmarks:
        # print(results.multi_handedness[0])
        handedness = results.multi_handedness
        hand_landmarks = results.multi_hand_landmarks
        Hands_List=[]
        Hands_Box_List=[]
        
        checkbool=False

        if(len(hand_landmarks)==1):
            checkbool=True

        for i in range(len(hand_landmarks)):
            all_x = []
            all_y = []

            all_point=[]
        
            for hnd in mp_hands.HandLandmark:
                all_x.append(hand_landmarks[i].landmark[hnd].x)
                all_y.append(hand_landmarks[i].landmark[hnd].y)
                
                all_point.append([hand_landmarks[i].landmark[hnd].x* img.shape[1]+min_x
                                ,hand_landmarks[i].landmark[hnd].y* img.shape[0]+min_y,hnd])

            hand_x1 = int(np.min(all_x) * img.shape[1] + min_x)
            hand_y1 = int(np.min(all_y) * img.shape[0] + min_y)
            hand_x2 = int(np.max(all_x) * img.shape[1] + min_x)
            hand_y2 = int(np.max(all_y) * img.shape[0] + min_y)

            hand_x1 = 0 if hand_x1 - 10 < 0 else hand_x1 - 10
            hand_y1 = 0 if hand_y1 - 10 < 0 else hand_y1 - 10
            hand_x2 = width - 1 if hand_x2 + 10 >= width else hand_x2 + 10
            hand_y2 = height - 1 if hand_y2 + 10 >= height else hand_y2 + 10

            wrist_x, wrist_y = hand_landmarks[i].landmark[mp_hands.HandLandmark.WRIST].x, hand_landmarks[i].landmark[mp_hands.HandLandmark.WRIST].y


            x, y = int(wrist_x * img.shape[1] + min_x), int(wrist_y * img.shape[0] + min_y)
            wrists.append([x, y])



            # print(handedness[i].classification[0].label)
            hand = handedness[i].classification[0].label

            
            
            

            
            Hands_List.append(all_point)
            Hands_Box_List.append([hand_x1,hand_y1,hand_x2,hand_y2])


        finger =['Thum','Index','Middle','Ring','Pinky']

       

        
        if(checkbool):

            if str(hand)=='Left':
                dir='R'
            else:
                dir='L'

            label_json['shapes'].append({
                "label": f'{p_prefix}_'+f'{dir}'+'H',
                "points": [
                    (hand_x1, hand_y1), 
                    (hand_x2, hand_y2)
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            })

            

            for i in range(0,5):
                label_json['shapes'].append({
                    "label": f'{p_prefix}_'+f'{dir}'+'H_'+f'{finger[i]}'+'_skel',
                    "points": [
                        #[basePoint_R[0],basePoint_R[1]],
                        [all_point[i*4+1][0],all_point[i*4+1][1]],
                        [all_point[i*4+2][0],all_point[i*4+2][1]],
                        [all_point[i*4+3][0],all_point[i*4+3][1]],
                        [all_point[i*4+4][0],all_point[i*4+4][1]]
                    ],
                    "group_id": None,
                    "description": "",
                    "shape_type": "linestrip",
                    "flags": {}
                })
        else:
            RH_IND,LH_IND=RHCheck(basePoint_R,basePoint_L,Hands_List)

            
            label_json['shapes'].append({
                "label": f'{p_prefix}_LH',
                "points": [
                    (Hands_Box_List[RH_IND][0], Hands_Box_List[RH_IND][1]), 
                    (Hands_Box_List[RH_IND][2], Hands_Box_List[RH_IND][3])
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            })

            label_json['shapes'].append({
                "label": f'{p_prefix}_RH',
                "points": [
                    (Hands_Box_List[LH_IND][0], Hands_Box_List[LH_IND][1]), 
                    (Hands_Box_List[LH_IND][2], Hands_Box_List[LH_IND][3])
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            })

            for i in range(0,5):
                label_json['shapes'].append({
                    "label": f'{p_prefix}_'+'LH_'+f'{finger[i]}'+'_skel',
                    "points": [
                        #[basePoint_R[0],basePoint_R[1]],
                        [Hands_List[RH_IND][i*4+1][0],Hands_List[RH_IND][i*4+1][1]],
                        [Hands_List[RH_IND][i*4+2][0],Hands_List[RH_IND][i*4+2][1]],
                        [Hands_List[RH_IND][i*4+3][0],Hands_List[RH_IND][i*4+3][1]],
                        [Hands_List[RH_IND][i*4+4][0],Hands_List[RH_IND][i*4+4][1]]
                    ],
                    "group_id": None,
                    "description": "",
                    "shape_type": "linestrip",
                    "flags": {}
                })

            for i in range(0,5):
                label_json['shapes'].append({
                    "label": f'{p_prefix}_'+'RH_'+f'{finger[i]}'+'_skel',
                    "points": [
                        #[basePoint_L[0],basePoint_L[1]],
                        [Hands_List[LH_IND][i*4+1][0],Hands_List[LH_IND][i*4+1][1]],
                        [Hands_List[LH_IND][i*4+2][0],Hands_List[LH_IND][i*4+2][1]],
                        [Hands_List[LH_IND][i*4+3][0],Hands_List[LH_IND][i*4+3][1]],
                        [Hands_List[LH_IND][i*4+4][0],Hands_List[LH_IND][i*4+4][1]]
                    ],
                    "group_id": None,
                    "description": "",
                    "shape_type": "linestrip",
                    "flags": {}
                })




def allfile_pose_detection(imageList):

    for file in imageList:
        pose_detection(image_path=file)


if __name__ == "__main__":
    pose_detection('./pose_test/[004][Top4K]20230524_180000(20230616153801)_194.jpg')
