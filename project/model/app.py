from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
import torch
from torchvision import transforms
from PIL import Image
from flask_cors import CORS  # 解决跨域问题
from model import AlexNet  # 假设你已经实现了 AlexNet 模型

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)  # 启用跨域请求支持

# 配置 SQLite 数据库
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SECRET_KEY'] = 'your_secret_key'  # 用于 session 和 flash 消息
db = SQLAlchemy(app)

# 用户模型
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    comments = db.relationship('Comment', backref='author', lazy=True)

# 评论模型
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(500), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# 加载模型
model = AlexNet()
model.load_state_dict(torch.load('D:\测试\pythonProject\project\cat_dog_website\model\Alexnet.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

# 定义图片预处理步骤
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 首页路由
@app.route('/')
def index():
    return render_template('home.html')

# 用户注册路由
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        flash('注册成功，请登录！', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

# 用户登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id  # 将用户 ID 存入 session
            flash('登录成功！', 'success')
            return redirect(url_for('index'))
        else:
            flash('登录失败，请检查用户名或密码！', 'danger')
    return render_template('login.html')

# 用户注销路由
@app.route('/logout')
def logout():
    session.pop('user_id', None)  # 清除 session 中的用户 ID
    flash('您已成功注销！', 'success')
    return redirect(url_for('index'))

# 图片上传和预测路由
@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:  # 检查用户是否登录
        flash('请先登录以进行猫狗检测！', 'danger')
        return redirect(url_for('login'))

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

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

# 评论页面路由
@app.route('/comments', methods=['GET', 'POST'])
def comments():
    if 'user_id' not in session:  # 检查用户是否登录
        flash('请先登录以发表评论！', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        content = request.form['comment']
        user_id = session['user_id']  # 从 session 中获取当前用户 ID
        comment = Comment(content=content, user_id=user_id)
        db.session.add(comment)
        db.session.commit()
        flash('评论已发布！', 'success')
    all_comments = Comment.query.all()
    return render_template('comments.html', comments=all_comments)

# 初始化数据库
def initialize_database():
    with app.app_context():
        db.create_all()

# 启动 Flask 应用
if __name__ == "__main__":
    initialize_database()  # 初始化数据库
    app.run(host='0.0.0.0', port=5000, debug=True)