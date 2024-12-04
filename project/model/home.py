from flask import Flask, request, jsonify, render_template
import torch
from torchvision import models, transforms
from PIL import Image
from flask_cors import CORS  # 解决跨域问题
from model import AlexNet

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)  # 启用跨域请求支持

# 加载模型
model = AlexNet()
model.load_state_dict(torch.load('D:/测试/pythonProject/project/model/Alexnet.pth'))
model.eval()

# 定义预处理步骤
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 主页路由，渲染index.html
@app.route('/')
def index():
    return render_template('home.html')

# 预测路由
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    print(f"Received file: {file.filename}")  # 用于调试

    try:
        img = Image.open(file.stream)
        img = transform(img).unsqueeze(0)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {e}'})

    try:
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            result = "猫" if predicted.item() == 0 else "狗"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': f'Error in model prediction: {e}'})

# 启动 Flask 应用
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000,debug=True)
