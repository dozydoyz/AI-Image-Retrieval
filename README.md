# 基于 DINOv2 与 ResNet50 的 AI 以图搜图系统

一款基于Django框架构建的智能图像检索与识别系统。融合了DINOv2（用于高精度特征提取与相似度检索）与ResNet50（用于添加Label）并支持CPU(NumPy)与GPU(CuPy)双模式推理，实现了从上传、特征提取到结果展示的效果

## 目录
- 简介
- 功能特性
- 运行环境与依赖
- 关于图库索引与自定义数据
- 运行流程说明
- 项目结构

## 功能特性

- 提供两个训练方式
  
  CPU模式：基于NumPy实现，无需显卡即可运行，适合轻量级部署（默认开启）。
  
  GPU模式：基于CuPy实现矩阵运算加速，显著加快训练速度。

- 智能结果展示
  
  Top-10相似推荐：采用余弦相似度计算，按相似分值降序排列，网格化展示检索结果。多维输出可能的Label。

- 历史记录管理
  
  基于Django实现用户搜索历史的持久化存储，支持随时回溯与再次搜索。

## 运行环境与依赖

使用Python 3.11环境以获得最佳兼容性，**不兼容Python>3.11的版本**

所需基础依赖：

- Python 3.11
- Django >= 4.0
- torch
- torchvision
- numpy
- Pillow
- tqdm
- django-bootstrap4

GPU加速依赖（可选）：

- cupy-cuda11x 或 cupy-cuda12x (根据你的 CUDA 版本选择)

安装依赖：

```bash
pip install -r requirements.txt
```

关于 GPU 加速的特别说明

`requirements.txt` 默认仅包含CPU运行所需的库。若需开启GPU加速，请先检查本地CUDA版本（使用 `nvcc --version` 命令），然后手动安装对应的 CuPy 版本。
 例如CUDA 12.x用户请执行：
```
pip install cupy-cuda12x
```

## 关于图库索引与自定义数据

本项目核心依赖于`.npz`格式的特征索引文件。你可以选择直接使用预训练好的索引，也可以根据自己的图片构建新索引。

### 方式一：使用预置索引

项目`core`目录下默认包含一个`gallery_features.npz`文件。

* **数据规模**：包含约 40,000 张已处理的图像特征。
* **使用方法**：直接运行项目即可。系统会自动加载该文件，无需任何额外训练。

### 方式二：构建自定义图库（训练自己的数据）

如果你想让系统搜索你自己的图片文件夹，请按以下步骤操作：

1. **准备图片**
   在项目根目录下创建`static\gallery_images`文件夹（或在`settings.py`中配置的静态目录），再将待训练图片导入。

2. **运行构建脚本**
   根据你的硬件条件，运行对应的索引构建脚本。脚本会自动扫描文件夹，使用DINOv2提取特征并保存为`.npz`文件。

   **CPU用户运行：**
   ```bash
   python -m core.buildindex_numpy
   ```
   **GPU用户运行：**
   ```bash
   python -m core.buildindex_gpu
   ```
3. **重启服务**
   
   新生成的gallery_features.npz会覆盖旧文件，重启Django服务器即可生效。

## 运行流程说明

程序启动前请确保已准备好模型权重文件与图库索引。

### 1. 克隆项目与环境准备

建立并激活 **Python 3.11** 虚拟环境，安装上述依赖。

### 2. 检查模型文件


确保 `core` 目录下存在以下两个 `.npz` 文件（若缺失需手动下载放置）：

* `vit-dinov2-base.npz` (DINOv2 权重)
* `gallery_features.npz` (图库特征索引)

### 3. 数据库迁移

初始化 Django 数据库（SQLite）：

```bash
python manage.py makemigrations
python manage.py migrate
```

### 4. 启动服务器

运行以下命令启动 Web 服务：

```bash
python manage.py runserver
```
服务启动后，访问 http://127.0.0.1:8000 即可进入系统。

## 项目结构
```
project/
│── manage.py               # Django 项目入口
│── core/                   # 核心应用目录
│   │── views.py            # 视图逻辑
│   │── models.py           # 数据库模型
│   │── forms.py            # 表单定义
│   │── urls.py             # 路由配置
│   │── services.py         # 搜图引擎主入口
│   │── classifier.py       # 认图引擎
│   │── preprocess_image.py # 图像预处理
│   │── dinov2_numpy.py     # DINOv2模型CPU推理实现
│   │── dinov2_gpu.py       # DINOv2模型GPU推理实现 (基于CuPy)
│   │── buildindex_numpy.py # [工具]CPU版本特征库构建脚本
│   │── buildindex_gpu.py   # [工具]GPU版本特征库构建脚本
│   │── download.py         # [工具]模型权重下载脚本
│   │── vit-dinov2-base.npz # DINOv2 模型权重文件
│   └── gallery_features.npz# 图库特征索引文件
│── templates/              # HTML 模板
│   └── core/
│       │── index.html      # 首页上传
│       │── results.html    # 结果展示
│       └── ...
│── static/                 # 静态文件
│── requirements.txt        # 依赖列表
└── README.md               # 项目说明文档
