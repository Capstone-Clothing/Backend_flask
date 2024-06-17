from flask import Flask, request, jsonify
import boto3
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np

app = Flask(__name__)

HSV = {
    (213 / 2, 4.3 * (255 / 100), 100 * (255 / 100)): '깨끗한 흰색',
    (150 / 2, 1.0 * (255 / 100), 80.0 * (255 / 100)): '조용한 연한 회색',
    (60 / 2, 0.6 * (255 / 100), 63.5 * (255 / 100)): '조용한 회색',
    (249 / 2, 8.5 * (255 / 100), 64.3 * (255 / 100)): '조용한 보라빛 회색',
    (325 / 2, 19.9 * (255 / 100), 76.9 * (255 / 100)): '조용한 분홍빛 회색',
    (203 / 2, 7.9 * (255 / 100), 64.7 * (255 / 100)): '조용한 파란빛 회색',
    (50 / 2, 7.8 * (255 / 100), 30.2 * (255 / 100)): '조용한 갈색빛 회색',
    (228 / 2, 4.9 * (255 / 100), 40.4 * (255 / 100)): '조용한 어두운 회색',
    (240 / 2, 7.1 * (255 / 100), 16.5 * (255 / 100)): '깜깜한 검정색',

    (40 / 2, 18.1 * (255 / 100), 84.3 * (255 / 100)): '부드러운 베이지색',
    (45 / 2, 10.7 * (255 / 100), 88.2 * (255 / 100)): '부드러운 연한 베이지색',
    (44 / 2, 29.2 * (255 / 100), 95.3 * (255 / 100)): '부드러운 밝은 베이지색',
    (70 / 2, 18.7 * (255 / 100), 85.9 * (255 / 100)): '부드러운 연두빛 베이지색',
    (8 / 2, 35.0 * (255 / 100), 88.6 * (255 / 100)): '따스한 분홍빛 살구색',
    (32 / 2, 37.4 * (255 / 100), 89.0 * (255 / 100)): '따스한 살구색',
    (37 / 2, 53.5 * (255 / 100), 94.5 * (255 / 100)): '풍부한 연한 황토색',
    (29 / 2, 79.5 * (255 / 100), 78.4 * (255 / 100)): '풍부한 황토색',
    (31 / 2, 71.2 * (255 / 100), 60.0 * (255 / 100)): '풍부한 진한 황토색',

    (6 / 2, 45.9 * (255 / 100), 91.4 * (255 / 100)): '화사한 연한 다홍색',
    (6 / 2, 62.0 * (255 / 100), 91.8 * (255 / 100)): '화사한 다홍색',
    (6 / 2, 58.6 * (255 / 100), 77.6 * (255 / 100)): '화사한 어두운 다홍색',
    (359 / 2, 86.5 * (255 / 100), 81.6 * (255 / 100)): '열정적인 빨간색',
    (349 / 2, 94.1 * (255 / 100), 66.7 * (255 / 100)): '열정적인 어두운 빨간색',
    (337 / 2, 32.9 * (255 / 100), 94.1 * (255 / 100)): '사랑스러운 연한 분홍색',
    (338 / 2, 48.8 * (255 / 100), 94.1 * (255 / 100)): '사랑스러운 분홍색',
    (337 / 2, 59.9 * (255 / 100), 89.0 * (255 / 100)): '사랑스러운 밝은 분홍색',
    (330 / 2, 67.3 * (255 / 100), 79.2 * (255 / 100)): '사랑스러운 진한 분홍색',
    (333 / 2, 54.4 * (255 / 100), 76.5 * (255 / 100)): '사랑스러운 어두운 분홍색',

    (54 / 2, 50.0 * (255 / 100), 95.7 * (255 / 100)): '발랄한 연한 노란색',
    (54 / 2, 91.7 * (255 / 100), 99.6 * (255 / 100)): '발랄한 노란색',
    (50 / 2, 100.0 * (255 / 100), 100.0 * (255 / 100)): '발랄한 진한 노란색',
    (39 / 2, 100.0 * (255 / 100), 100.0 * (255 / 100)): '따뜻한 연한 주황색',
    (29 / 2, 98.4 * (255 / 100), 100.0 * (255 / 100)): '따뜻한 주황색',
    (27 / 2, 88.8 * (255 / 100), 76.9 * (255 / 100)): '따뜻한 어두운 주황색',
    (29 / 2, 69.6 * (255 / 100), 43.9 * (255 / 100)): '단단한 밝은 갈색',
    (4 / 2, 60.4 * (255 / 100), 41.6 * (255 / 100)): '단단한 갈색',
    (12 / 2, 47.7 * (255 / 100), 33.7 * (255 / 100)): '단단한 어두운 갈색',
    (355 / 2, 66.1 * (255 / 100), 45.1 * (255 / 100)): '단단한 붉은빛 갈색',

    (120 / 2, 39.8 * (255 / 100), 82.7 * (255 / 100)): '싱그러운 연한 연두색',
    (126 / 2, 66.1 * (255 / 100), 65.9 * (255 / 100)): '싱그러운 연두색',
    (58 / 2, 63.0 * (255 / 100), 60.4 * (255 / 100)): '싱그러운 갈색빛 연두색',
    (150 / 2, 100.0 * (255 / 100), 54.9 * (255 / 100)): '신선한 초록색',
    (132 / 2, 61.7 * (255 / 100), 31.8 * (255 / 100)): '신선한 어두운 초록색',
    (48 / 2, 97.1 * (255 / 100), 66.7 * (255 / 100)): '신선한 갈색빛 초록색',
    (184 / 2, 30.3 * (255 / 100), 89.4 * (255 / 100)): '상쾌한 연한 청록색',
    (172 / 2, 100.0 * (255 / 100), 72.2 * (255 / 100)): '상쾌한 밝은 청록색',
    (167 / 2, 100.0 * (255 / 100), 58.0 * (255 / 100)): '상쾌한 청록색',
    (183 / 2, 66.3 * (255 / 100), 36.1 * (255 / 100)): '상쾌한 어두운 청록색',

    (196 / 2, 41.1 * (255 / 100), 90.6 * (255 / 100)): '맑은 연한 하늘색',
    (189 / 2, 100.0 * (255 / 100), 82.4 * (255 / 100)): '맑은 하늘색',
    (208 / 2, 100.0 * (255 / 100), 61.2 * (255 / 100)): '시원한 파란색',
    (222 / 2, 74.8 * (255 / 100), 49.8 * (255 / 100)): '시원한 어두운 파란색',
    (233 / 2, 53.8 * (255 / 100), 31.4 * (255 / 100)): '단정한 남색',

    (327 / 2, 66.7 * (255 / 100), 42.4 * (255 / 100)): '우아한 자주색',
    (322 / 2, 66.4 * (255 / 100), 57.3 * (255 / 100)): '우아한 밝은 자주색',
    (263 / 2, 29.2 * (255 / 100), 72.5 * (255 / 100)): '신비로운 연한 보라색',
    (314 / 2, 61.5 * (255 / 100), 51.0 * (255 / 100)): '신비로운 보라색',
    (319 / 2, 62.9 * (255 / 100), 34.9 * (255 / 100)): '신비로운 어두운 보라색',
}
hsv = list(HSV.keys())
hsv_len = len(hsv)


if not app.debug:
    handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)

def download_image_from_s3(bucket_name, key):
    s3 = boto3.client('s3')
    app.logger.info(f"Attempting to download image from bucket: {bucket_name}, key: {key}")
    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        image_data = response['Body'].read()
        image = Image.open(BytesIO(image_data))
        return image
    except s3.exceptions.NoSuchKey:
        app.logger.error(f"NoSuchKey error: Bucket: {bucket_name}, Key: {key}")
        raise

def preprocess_image(image, target_size):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(18432, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model_path, num_classes):
    model = CNNModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(image, model, device):
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
    return outputs

def decode_predictions(predictions, class_labels):
    _, max_index = torch.max(predictions, 1)
    predicted_label = class_labels[max_index.item()]
    return predicted_label

def extract_color(rgb_list):
    for i in range(len(rgb_list)):
        rgb_list[i] = np.uint8([[rgb_list[i]]])
        rgb_list[i] = cv2.cvtColor(rgb_list[i], cv2.COLOR_RGB2HSV)
        rgb_list[i] = rgb_list[i][0][0]

    color_name_list = []
    for i in range(len(rgb_list)):
        minimum = float('inf')
        closest_color = None
        for j in range(hsv_len):
            chai = sum(abs(rgb_list[i][k] - hsv[j][k]) * (3 - k) for k in range(3))
            if chai < minimum:
                minimum = chai
                closest_color = HSV[hsv[j]]
        color_name_list.append(closest_color)
    return color_name_list

pattern_model_path = "./fashion_classifier_model_p.pth"
type_model_path = "./fashion_classifier_model.pth"
type_classes = ['재킷', '조거팬츠', '짚업', '스커트', '가디건', '점퍼', '티셔츠', '셔츠', '팬츠', '드레스', '패딩', '청바지', '점프수트', '니트웨어', '베스트', '코트', '브라탑', '블라우스', '탑', '후드티', '래깅스']
pattern_classes = ['페이즐리', '하트', '지그재그', '깅엄', '하운즈 투스', '도트', '레터링', '믹스', '뱀피', '해골', '체크', '무지', '카무플라쥬', '그라데이션', '스트라이프', '호피', '아가일', '그래픽', '지브라', '타이다이', '플로럴']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pattern_model = load_model(pattern_model_path, len(pattern_classes))
type_model = load_model(type_model_path, len(type_classes))
pattern_model = pattern_model.to(device)
type_model = type_model.to(device)

@app.route('/predict', methods=['POST'])
def predict_api():
    app.logger.info("Received request: %s", request.json)
    try:
        data = request.json
        bucket_name = data['codinaviImage']
        key = data['key']
        
        app.logger.info(f"Received request to download image from S3 - Bucket: {bucket_name}, Key: {key}")
        image = download_image_from_s3(bucket_name, key)
        app.logger.info("Image downloaded successfully")

        target_size = (224, 224)
        preprocessed_image = preprocess_image(image, target_size)
        app.logger.info("Image preprocessed successfully")

        pattern_predictions = predict(preprocessed_image, pattern_model, device)
        type_predictions = predict(preprocessed_image, type_model, device)
        app.logger.info("Prediction successful")

        predicted_pattern = decode_predictions(pattern_predictions, pattern_classes)
        predicted_type = decode_predictions(type_predictions, type_classes)

        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        pixels = np.float32(image.reshape(-1, 3))
        n_colors = 5
        _, labels, palette = cv2.kmeans(pixels, n_colors, None,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 10,
                                        cv2.KMEANS_RANDOM_CENTERS)
        _, counts = np.unique(labels, return_counts=True)
        dominant_color = palette[np.argmax(counts)]
        dominant_color_name = extract_color([dominant_color])[0]

        result = {
            "pattern": predicted_pattern,
            "type": predicted_type,
            "dominant_color": dominant_color_name
        }

        response = json.dumps(result, ensure_ascii=False)
        app.logger.info("Response: %s", response)
        return Response(response, content_type='application/json')
    except Exception as e:
        app.logger.error(f"Error in /predict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
