import cv2
import mediapipe as mp
import numpy as np
import math
import streamlit as st

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Hand Gesture Canvas")
run = st.checkbox("Start Camera")
frame_window = st.empty()

# -------------------------------
# Camera Setup
# -------------------------------
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# -------------------------------
# MediaPipe Setup
# -------------------------------
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(min_detection_confidence=0.75,
                                min_tracking_confidence=0.75)

mp_draw = mp.solutions.drawing_utils

# -------------------------------
# Canvas
# -------------------------------
canvas = np.zeros((720,1280,3),np.uint8)

# -------------------------------
# Palette Settings
# -------------------------------
palette_height = 100
box_size = 60
gap = 15
start_x = 20

tool_width = 110
tool_height = 60

colors = [
(0,0,255),
(0,255,255),
(255,255,0),
(0,255,0),
(147,20,255),
(238,130,238),
(255,255,255)
]

current_color = (0,0,255)
current_tool = "pen"

brush_thickness = 6
neon_thickness = 8
eraser_thickness = 40

xp,yp = 0,0

PINCH_NORM_THRESHOLD = 0.02
PINCH_PIXEL_MIN = 10
PINCH_HOLD_FRAMES = 3

pinch_frames = 0
prev_fist = False

# -------------------------------
# Rounded Rectangle
# -------------------------------
def rounded_rect(img,x,y,w,h,color):
    r=12
    cv2.rectangle(img,(x+r,y),(x+w-r,y+h),color,-1)
    cv2.rectangle(img,(x,y+r),(x+w,y+h-r),color,-1)
    cv2.circle(img,(x+r,y+r),r,color,-1)
    cv2.circle(img,(x+w-r,y+r),r,color,-1)
    cv2.circle(img,(x+r,y+h-r),r,color,-1)
    cv2.circle(img,(x+w-r,y+h-r),r,color,-1)

# -------------------------------
# Draw Palette
# -------------------------------
def draw_palette(img):

    cv2.rectangle(img,(0,0),(1280,palette_height),(0,0,0),-1)

    x=start_x
    for c in colors:
        size = box_size
        if c == current_color:
            size += 8
            rounded_rect(img,x-4,16,size,size,c)
            cv2.rectangle(img,(x-6,14),(x+size-2,16+size),(255,255,255),2)
        rounded_rect(img,x,20,box_size,box_size,c)
        x += box_size + gap

    tool_color = (203,192,255)
    size_w = tool_width
    size_h = tool_height

    if current_tool=="pen":
        size_w += 10
        size_h += 6
        rounded_rect(img,x-5,17,size_w,size_h,tool_color)
        cv2.rectangle(img,(x-7,15),(x+size_w-3,17+size_h),(255,255,255),2)
    rounded_rect(img,x,20,tool_width,tool_height,tool_color)
    cv2.putText(img,"Pen",(x+30,55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,(0,0,0),2)
    pen_start = x
    pen_end = x + tool_width

    x += tool_width + gap
    if current_tool=="neon":
        size_w = tool_width + 10
        size_h = tool_height + 6
        rounded_rect(img,x-5,17,size_w,size_h,tool_color)
        cv2.rectangle(img,(x-7,15),(x+size_w-3,17+size_h),(255,255,255),2)
    rounded_rect(img,x,20,tool_width,tool_height,tool_color)
    cv2.putText(img,"Neon",(x+25,55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,(0,0,0),2)
    neon_start = x
    neon_end = x + tool_width

    return pen_start, pen_end, neon_start, neon_end

# -------------------------------
# Neon Glow
# -------------------------------
def neon(canvas_img,x1,y1,x2,y2,color,size):

    for i,alpha in enumerate([0.25,0.18,0.12],start=1):
        glow_size = size + i*4
        overlay = canvas_img.copy()
        cv2.line(overlay,(x1,y1),(x2,y2),color,glow_size,cv2.LINE_AA)
        cv2.addWeighted(overlay,alpha,canvas_img,1-alpha,0,canvas_img)

    core_size = max(2,size//3)
    cv2.line(canvas_img,(x1,y1),(x2,y2),(255,255,255),core_size,cv2.LINE_AA)

# -------------------------------
# Main Loop
# -------------------------------
while run:

    success,img = cap.read()
    if not success:
        break

    img = cv2.flip(img,1)

    pen_start, pen_end, neon_start, neon_end = draw_palette(img)

    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)

    cx,cy,tx,ty = 0,0,0,0

    if result.multi_hand_landmarks:

        hand_landmarks = result.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            img,hand_landmarks,mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(255,255,255),thickness=1,circle_radius=2),
            mp_draw.DrawingSpec(color=(255,255,255),thickness=1)
        )

        h,w,_ = img.shape
        lm = hand_landmarks.landmark

        index = lm[8]
        thumb = lm[4]

        cx,cy = int(index.x*w),int(index.y*h)
        tx,ty = int(thumb.x*w),int(thumb.y*h)

        pinch = np.hypot(cx-tx,cy-ty)
        pinch_norm = pinch/float(w)

        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up = lm[16].y < lm[14].y
        pinky_up = lm[20].y < lm[18].y

        ring_down = lm[16].y > lm[14].y
        pinky_down = lm[20].y > lm[18].y

        is_pinch = (pinch_norm<PINCH_NORM_THRESHOLD) or (pinch<PINCH_PIXEL_MIN)

        fist = not(index_up or middle_up or ring_up or pinky_up)
        open_palm = index_up and middle_up and ring_up and pinky_up

        if prev_fist and open_palm:
            canvas = np.zeros((720,1280,3),np.uint8)

        prev_fist = fist

        erasing = index_up and middle_up and ring_down and pinky_down and pinch_frames==0

        if erasing:

            if xp==0 and yp==0:
                xp,yp=cx,cy

            cv2.line(canvas,(xp,yp),(cx,cy),(0,0,0),eraser_thickness)
            xp,yp=cx,cy

        else:

            if is_pinch:
                pinch_frames+=1
            else:
                pinch_frames=0

            if cy < palette_height:

                x=start_x

                for c in colors:
                    if x < cx < x+box_size:
                        current_color=c
                    x += box_size + gap

                if pen_start < cx < pen_end:
                    current_tool="pen"

                if neon_start < cx < neon_end:
                    current_tool="neon"

            else:

                if pinch_frames >= PINCH_HOLD_FRAMES:

                    if xp==0 and yp==0:
                        xp,yp=cx,cy

                    dx = cx - xp
                    dy = cy - yp
                    distance = int(math.hypot(dx, dy))
                    if distance == 0:
                        distance = 1

                    for i in range(distance):
                        x = int(xp + dx * i / distance)
                        y = int(yp + dy * i / distance)

                        if current_tool=="pen":
                            cv2.circle(canvas,(x,y),
                                       brush_thickness//2,
                                       current_color,-1)

                    if current_tool=="neon":
                        neon(canvas,xp,yp,cx,cy,current_color,neon_thickness)

                    xp,yp=cx,cy
                else:
                    xp,yp=0,0

    else:
        pinch_frames=0
        xp,yp=0,0

    if cx!=0 or cy!=0:
        cv2.circle(img,(cx,cy),6,(255,255,255),-1)

    imgGray=cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    _,imgInv=cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)

    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,canvas)

    frame_window.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))