import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH='models/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker= vision.HandLandmarker
HandLandmarkerOptions= vision.HandLandmarkerOptions
VisionRunningMode= vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands = 2
)

landmarker = HandLandmarker.create_from_options(options)

cam = cv2.VideoCapture(0)
while cam.isOpened:
    success , frame = cam.read()
    if success == False:
        break
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frameRGB)
    timestamp_ms = int(cam.get(cv2.CAP_PROP_POS_MSEC))
    result = landmarker.detect_for_video(mp_image,timestamp_ms)
    
    
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                h, w, _ = frame.shape
                x, y = int(lm.x*w) , int(lm.y*h)
                cv2.circle(frame,(x,y),5,(0,0,255),-1)


    cv2.imshow('window',frame)
    # cv2.imshow('RGB',frameRGB)
    print(result.hand_landmarks)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()




# [[NormalizedLandmark(x=0.4816787838935852, y=0.8062804341316223, z=3.582246108635445e-07, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.42766207456588745, y=0.7575019598007202, z=-0.010562509298324585, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.3956836462020874, y=0.6842576265335083, z=-0.023002827540040016, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.37333476543426514, y=0.6300444602966309, z=-0.03623022139072418, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.34663063287734985, y=0.5861279368400574, z=-0.0518924817442894, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.4297455847263336, y=0.600828230381012, z=-0.036455363035202026, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.41601866483688354, y=0.5107881426811218, z=-0.06603740900754929, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.39659348130226135, y=0.4612085223197937, z=-0.08656994998455048, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.373026043176651, y=0.4185795783996582, z=-0.10146516561508179, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.4501582384109497, y=0.6086929440498352, z=-0.0529024712741375, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.4390849769115448, y=0.5054516792297363, z=-0.08392561972141266, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.4226965606212616, y=0.43619444966316223, z=-0.10698620975017548, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.4046690762042999, y=0.37569043040275574, z=-0.12277987599372864, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.4714234471321106, y=0.6385676264762878, z=-0.070098377764225, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.4829050302505493, y=0.5400621891021729, z=-0.1038309782743454, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.48499250411987305, y=0.47584736347198486, z=-0.12557227909564972, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.48204609751701355, y=0.4158266484737396, z=-0.13979320228099823, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.49160248041152954, y=0.6860515475273132, z=-0.08682319521903992, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.5147565603256226, y=0.6302103400230408, z=-0.11969012767076492, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.5266949534416199, y=0.5896861553192139, z=-0.13409414887428284, visibility=None, presence=None, name=None), NormalizedLandmark(x=0.5351426601409912, y=0.5489630103111267, z=-0.1431201845407486, visibility=None, presence=None, name=None)]]