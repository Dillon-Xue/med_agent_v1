FROM python:3.10-slim

WORKDIR /app

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .
COPY . .
COPY utils/database.py /app/utils/database.py


RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

RUN pip install DBUtils


# 构建阶段运行测试
ENV DASHSCOPE_API_KEY=test-key
ENV LLM_PROVIDER=dashscope
# RUN PYTHONPATH=. python -m pytest tests/ -v --tb=short

EXPOSE 8000

CMD ["uvicorn", "chat:app", "--host", "0.0.0.0", "--port", "8000"]