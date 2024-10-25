# 选择基础镜像
FROM python:3.10


# 设置工作目录
WORKDIR /app

# 复制当前目录下的文件到容器中的工作目录
COPY . /app

# 安装所需的依赖
RUN pip install --no-cache-dir -r requirements.txt

# 指定运行命令
CMD ["python", "app.py"]
