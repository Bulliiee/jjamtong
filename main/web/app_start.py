from gc import callbacks
from socket import socket
from flask import Flask, render_template, Response, request
import urllib.request
import cv2
from web import FCMManager
from flask_socketio import SocketIO,emit

#Test device Token
tokens1 = ["fu0WOWGkTTak56YUfenfLC:APA91bFG5nHql7K2xlQ8DpxU1QLhDGXRPEc4NdABvNahEjCSIvzAuNGmx4AyWl_yWKqmdURmlrwlW0BLHYcf6jdS8R5QK4FRYXB_yq8yBhoiudAIa3zzd3s3eMQulTPQSNpVz16DCGAS"]

#Flask 초기 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = '1111'
app.config['DEBUG']=True
socketio = SocketIO(app, logger = True, engineio_logger=True)

global capacity_value
capacity_value = ""

global init_value
init_value = ""

#img_src = cv2.imread("./web/static/")

#socket 통신 코드
@socketio.on('connect')
def test_connect():
    print("\n\n connet")
    
@socketio.on('disconnect')
def test_connect():
    print("\n\n disconnet")

#저장 용량에 대한 소켓 데이터
@socketio.on('capacity')
def handle_message(msg):
    global capacity_value
    capacity_value = str(msg)
    print("\n\n Message recieved " + msg)
    # push message send
    send_push_message()
    

#라즈베리파이 초기 코드에 대한 소켓 데이터
@socketio.on('init')
def handle_message(msg):
    global init_value 
    init_value = str(msg)
    print("\n\ninit value" + msg)

#Web page url link mapping
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/monitoring")
def monitoring():
    return render_template('moniter_main.html')

@app.route("/status")
def status():
    return render_template('status.html')

@app.route("/monitoring/capacity")
def capacity():
    print("\n\n capacity : " +capacity_value)
    #소켓 통신으로 받아온 값을 split 해서 list로 저장
    temp_value = capacity_value.split()
    print(temp_value)
    
    try:
        #리스트에 저장된 값 각각의 변수에 저장
        #캔 아래, 위
        can1 = int(temp_value[1])
        can2 = int(temp_value[0])
        #페트 아래, 위
        pet1 = int(temp_value[3])
        pet2 = int(temp_value[2])
        #일반 아래, 위
        gen1 = int(temp_value[5])
        gen2 = int(temp_value[4])
    except:
        #temp_value가 비어 있어 값을 못 넣을 때 예외 처리
        can1 = None
        can2 = None
        pet1 = None
        pet2 = None
        gen1 = None
        gen2 = None
    
    #초기 이상이 있을 경우 or 웹 서버와 라즈베리파이 소켓 통신이 되지 않아 값을 못 받아 올 때.
    if(can1 is None and can2 is None and pet1 is None and pet2 is None and gen1 is None and gen2 is None):
        can_capacity = "식별 불가"
        pet_capacity = "식별 불가"
        gen_capacity = "식별 불가"
    else:
        #초음파 센서 값을 받아와야 함.
        #각각의 센서 값을 받아와서 계산 후 용량을 %로 나타냄.
        #can값이 정상적으로 들어와 있을 때
        if(can1 <= 15):
            if(can2 <= 15):
                can_capacity = "30% 남았습니다."
            else:
                can_capacity = "70% 남았습니다."
        else:
            can_capacity = "70% 이상 남았습니다."

        #pet값이 정상적으로 들어와 있을 때
        if(pet1 <= 15):
            if(pet2 <= 15):
                pet_capacity = "30% 남았습니다."
            else:
                pet_capacity = "70% 남았습니다."
        else:
            pet_capacity = "70% 이상 남았습니다."
        
        #일반쓰래기 값이 정상적으로 들어와 있을 때
        if(gen1 <= 15):
            if(gen2 <= 15):
                gen_capacity = "30% 남았습니다."
            else:
                gen_capacity = "70% 남았습니다."
        else:
            gen_capacity = "70% 이상 남았습니다."

    return render_template('capacity.html' , can_capacity = can_capacity, pet_capacity = pet_capacity, gen_capacity = gen_capacity)

@app.route("/monitoring/stream")
def stream():
    return render_template('stream.html')


def send_push_message():
    print("\npush message call")
    temp_value = capacity_value.split()

    try:
        #리스트에 저장된 값 각각의 변수에 저장
        #캔 아래, 위
        print("\ntry")
        can1 = int(temp_value[1])
        can2 = int(temp_value[0])
        #페트 아래, 위
        pet1 = int(temp_value[3])
        pet2 = int(temp_value[2])
        #일반 아래, 위
        gen1 = int(temp_value[5])
        gen2 = int(temp_value[4])
    except:
        #temp_value가 비어 있어 값을 못 넣을 때 예외 처리
        print("\n\n except")
        can1 = None
        can2 = None
        pet1 = None
        pet2 = None
        gen1 = None
        gen2 = None
    
    #초기 이상이 있을 경우 or 웹 서버와 라즈베리파이 소켓 통신이 되지 않아 값을 못 받아 올 때.
    if(can1 is not None and can2 is not None and pet1 is not None and pet2 is not None and gen1 is not None and gen2 is not None):
        if(can1 <= 15):
            if(can2 <= 15):
                FCMManager.sendPush("중요!", "캔 잔여 용량이 얼마 남지 않았습니다. 확인 후 비워주세요!!", tokens1)

        #pet값이 정상적으로 들어와 있을 때
        if(pet1 <= 15):
            if(pet2 <= 15):
                FCMManager.sendPush("중요!", "페트 잔여 용량이 얼마 남지 않았습니다. 확인 후 비워주세요!!", tokens1)

        #일반쓰래기 값이 정상적으로 들어와 있을 때
        if(gen1 <= 15):
            if(gen2 <= 15):
                FCMManager.sendPush("중요!", "일반쓰레기 잔여 용량이 얼마 남지 않았습니다. 확인 후 비워주세요!!", tokens1)

if __name__=='__main__':
    socketio.run(app)