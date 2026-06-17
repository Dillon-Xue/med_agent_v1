FROM python:3.10-slim

WORKDIR /app

# 使用清华镜像源加速 pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# 复制项目代码（排除 vector_db 和 __pycache__ 等）
COPY . .

EXPOSE 8000

CMD ["uvicorn", "chat:app", "--host", "0.0.0.0", "--port", "8000"] 
