from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
from model import AlexNet

app = Flask(__name__)

# 加载模型
model = AlexNet()
model.load_state_dict(torch.load('Alexnet.pth', weights_only=True))
model.eval()

# 类别名称
classes = ["猫", "狗"]  # 根据你的数据集调整

# 图像预处理
def preprocess_image(image_path):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.05, 0.052, 0.047])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # 添加批次维度
    return image

# 预测函数
def predict_image(image):
    with torch.no_grad():
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        return classes[pre_lab.item()]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取上传的文件
        file = request.files['file']
        if file:
            # 保存文件到临时位置
            file_path = f'temp/{file.filename}'
            file.save(file_path)

            # 预处理并预测
            image = preprocess_image(file_path)
            result = predict_image(image)
            return render_template('index.html', result=result)

    return render_template('home.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)