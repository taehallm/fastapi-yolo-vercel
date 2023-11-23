from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from os import getenv

import base64
import cv2
import numpy as np
import torch
import time
import json
from ultralytics import YOLO  # YOLOv8 모델 라이브러리

from condition_check import burpees, pull_up, cross_lunge, side_lateral_raise, barbell_squat, push_up 
from countings import count_burpees, count_pull_up, count_cross_lunge, count_side_lateral_raise, count_barbell_squat, count_push_up

check_fns = {0: burpees, 1: pull_up, 2: cross_lunge, 3: side_lateral_raise, 4: barbell_squat, 5: push_up}
count_fns = {0: count_burpees, 1: count_pull_up, 2: count_cross_lunge, 3: count_side_lateral_raise, 4: count_barbell_squat, 5: count_push_up}
exercise = {0: 'burpee', 1: 'pull_up', 2: 'cross_lunge', 3: 'side_lateral_raise', 4: 'barbell_squat', 5: 'push_up'}

app = FastAPI()
model = YOLO("./best28.pt")  # 파인튜닝한 YOLOv8 모델 경로

# HTML for the webcam access
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Webcam Stream</title>
    <style>
        #container {
            display: flex;
            flex-direction: row; /* 가로 방향으로 요소들을 배치 */
            justify-content: center;
            align-items: center;
        }
        #video, #canvas {
            margin-right: 10px; /* 요소들 사이의 여백 */
        }
        #keypoints {
            white-space: pre; /* 줄바꿈과 공백을 유지 */
        }
    </style>
</head>
<body>
    <div id="container">
        <video id="video" width="640" height="480" autoplay playsinline></video>
        <canvas id="canvas" width="640" height="480"></canvas>
    </div>
    <div id="keypoints" style="margin-top: 10px;"></div>
    <div>
        <select id="exerciseDropdown">
            <option value="push_up">Push Up</option>
            <option value="pull_up">Pull Up 2</option>
            <option value="barbell_squat">Barbell Squat 3</option>
            <option value="cross_lunge">Cross Lunge</option>
            <option value="burpee">Burpee Test</option>
            <option value="side_lateral_raise">Side Lateral Raise</option>
            <!-- Add more options as needed -->
        </select>
        <button onclick="sendExerciseType()">Submit Exercise Type</button>
    </div>

    <script>
        var video = document.getElementById('video');
        var canvas = document.getElementById('canvas');
        var context = canvas.getContext('2d');
        var keypointsDiv = document.getElementById('keypoints');
        var ws = new WebSocket('ws://localhost:8000/ws');
        var captureInterval = 200; // 0.2 second for more real-time feel

        ws.onopen = function() {
            console.log('WebSocket connection opened');
            startStreaming();
        };

        ws.onmessage = function(event) {
            var keypoints = JSON.parse(event.data).keypoints; // 서버로부터 받은 키포인트 저장
            console.log(keypoints)
            drawKeypoints(keypoints); // 키포인트 그리기
            updateKeypointsText(keypoints); // 키포인트 텍스트 업데이트
        };

        function startStreaming() {
            setInterval(captureFrame, captureInterval);
        }

        function captureFrame() {
            if (ws.readyState === WebSocket.OPEN) {
                context.drawImage(video, 0, 0, 640, 480);
                var data = canvas.toDataURL('image/jpeg');
                ws.send(data);
            } else {
                console.log("WebSocket is not open. Current state:", ws.readyState);
            }
        }

        function drawKeypoints(keypoints) {
            context.clearRect(0, 0, canvas.width, canvas.height);
            keypoints.forEach(function(point) {
                context.beginPath();
                context.arc(point[0], point[1], 5, 0, 2 * Math.PI);
                context.fillStyle = 'red';
                context.fill();
            });
        }

        function updateKeypointsText(keypoints) {
            var keypointsDiv = document.getElementById('keypoints');
            keypointsDiv.innerHTML = keypoints.map(function(point, index) {
                return 'Point ' + index + ': (' + point[0].toFixed(2) + ', ' + point[1].toFixed(2) + ')';
            }).join('<br>');
        }

        // Access the camera
        if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
                video.srcObject = stream;
                video.play();
            }).catch(function(error) {
                console.error('getUserMedia error:', error);
            });
        }

        function sendExerciseType() {
            var exerciseType = document.getElementById('exerciseDropdown').value;
            fetch('/exercise', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ exercise_type: exerciseType }),
            })
            .then(response => response.json())
            .then(data => {
                console.log('Response:', data);
                // Handle the response here
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    </script>
</body>
</html> 
"""

def process_exercise(exercise_type: str):
    # Your algorithm here
    # Returns some data based on the exercise type
    return exercise_data

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.post("/exercise")
async def receive_exercise_type(request: Request):
    try:
        data = await request.json()
        exercise_type = data['exercise_type']
        exercise_data = process_exercise(exercise_type)
        if not exercise_type:
            raise ValueError("Exercise type is required")
        # Process the exercise type as needed
        return exercise_data
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            img_data = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # YOLO 모델로 키포인트 검출
            try:
                start_time = time.time()
                with torch.no_grad():
                    results = model.predict(img, save=False, imgsz=640, conf=0.5, device='cpu', verbose=False)[0]
                kpts = results.keypoints.xy[0].cpu().numpy()
                elapsed_time = time.time() - start_time
                print(f"Model prediction time: {elapsed_time} seconds")
                await websocket.send_json({'keypoints': kpts.tolist()})
            except Exception as e:
                print("Error during model prediction:", e)
                break  # or handle the error differently
    except Exception as e:
        print("WebSocket connection error:", e)
    finally:
        await websocket.close()

if __name__== "__main__":
    port = int(getenv("PORT", 8000))
    uvicorn.run("app.api:app", host="0.0.0.0", port=port, reload=True)
    