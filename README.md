# OAE：遮挡后物体外观一致性评估

本项目研究生成视频中的目标物体在完全遮挡并重新出现后，外观是否保持一致。目前是单场景 pilot，使用预训练 MobileNetV3 Small 提取图像特征，再训练轻量逻辑回归分类器。

## 目录与文件

```text
oae/
├── README.md
├── .gitignore
├── code/
│   ├── train_test.ipynb
│   └── MobileNetV3Small.ipynb
└── data/
    ├── raw_data/
    │   ├── raw_data.xlsx
    │   ├── video.zip
    │   └── video/
    └── labeled_data/
        ├── 0_PASS/
        │   ├── 0_PASS.xlsx
        │   ├── Cxxxx.mp4
        │   └── img/Cxxxx/
        ├── 1_FAIL/
        │   ├── 1_FAIL.xlsx
        │   ├── Cxxxx.mp4
        │   └── img/Cxxxx/
        ├── 2_Uncertain/
        │   ├── 2_uncertain.xlsx
        │   ├── Cxxxx.mp4
        │   └── img/
        ├── img/Cxxxx/
        ├── train/train.xlsx
        └── test/test.xlsx
```

`Cxxxx` 表示视频 ID。上图列出主要数据结构，实际视频、图片与压缩包名称以本地文件为准。GitHub 中可见的数据范围取决于 Git 跟踪情况及 `.gitignore`。

| 文件或目录 | 用途 |
| --- | --- |
| `README.md` | 项目说明、数据结构和运行方法。 |
| `.gitignore` | 定义 Git 忽略的文件规则。 |
| `code/train_test.ipynb` | 读取 Pass、Fail 标注表，分别划分训练和测试数据，合并、打乱并保存 Excel；也包含图片路径生成代码。 |
| `code/MobileNetV3Small.ipynb` | 提取前后帧特征、训练分类器、预测并展示初步评价结果。 |
| `data/raw_data/raw_data.xlsx` | 原始候选视频标注汇总。 |
| `data/raw_data/video.zip`、`video/` | 视频压缩包、分批压缩包及解压后的原始视频。 |
| `data/labeled_data/0_PASS/` | 外观一致的视频、对应标注表和截图。标签为 `0`。 |
| `data/labeled_data/1_FAIL/` | 外观发生变化的视频、对应标注表和截图。标签为 `1`。 |
| `data/labeled_data/2_Uncertain/` | 无法可靠判断或不符合有效事件条件的视频及记录。标签为 `2`，目前不用于分类器训练。 |
| 各分类目录下的 `img/Cxxxx/` | 按视频 ID 整理的截图；缺少有效前后画面的样本可能只有部分截图。 |
| `2_Uncertain/img/采帧清单.csv` | 已提取截图的采帧记录。 |
| `2_Uncertain/img/视频处理清单.csv` | 视频级处理情况记录。 |
| `data/labeled_data/img/Cxxxx/` | 当前模型实际读取的统一图片目录。 |
| `data/labeled_data/train/train.xlsx` | 训练数据清单。重新保存到同一路径会覆盖原文件。 |
| `data/labeled_data/test/test.xlsx` | 测试数据清单。重新保存到同一路径会覆盖原文件。 |

`.ipynb_checkpoints/` 是 Jupyter 自动检查点；`.DS_Store` 是 macOS 目录元数据；`.Rhistory` 是 R 历史记录文件。这些不属于模型流程。

## 标签与输入数据

- **Pass（0）**：遮挡前后目标外观保持一致。
- **Fail（1）**：遮挡后存在可确认的外观变化。
- **人工 Uncertain（2）**：证据不足或缺少有效遮挡事件。该目录大部分样本没有符合要求的遮挡事件，目前未用于训练；应按实际原因审核，而非将整个目录视为可训练的第三类。

当前分类器学习 `0` 和 `1`；预测阶段也可能输出 `2`，表示模型拒判，与数据中的人工 Uncertain 含义不同。

Excel 中模型直接使用的列为：

| 列 | 含义 |
| --- | --- |
| `ID` | 视频编号，例如 `C0005`。 |
| `consistency` | 人工类别标签。 |
| `before_1`、`before_2`、`before_3` | 遮挡前三张输入图像的路径。 |
| `after_1`、`after_2`、`after_3` | 重现后三张输入图像的路径。 |

标注表还可包含 `reason`、`color`、`shape`、`occlusion`、`reappearance_time` 等字段，用于记录判断原因、属性变化和事件时间。当前分类器不直接使用这些文本及时间字段。

路径示例：`/img/C0005/before_1.png`。代码将它拼接到 `../data/labeled_data` 后读取，因此它是项目内约定的路径片段，不是操作系统根目录下的绝对路径。

## 代码流程

### 1. `train_test.ipynb`：数据准备与划分

主要逻辑：

1. 使用 pandas 读取 Pass、Fail 的 Excel。
2. 对两个类别分别进行 80% / 20% 随机划分，固定 `random_state=42`。
3. 合并各类别的训练部分及测试部分，并打乱行顺序。
4. 通过 `save_or_update()` 创建目标目录并写出 Excel。该函数覆盖整个目标 Excel，不追加数据。
5. 根据每条记录自身的 ID 生成六张图片的路径。

**当前保存版本的执行顺序注意事项：** 路径生成单元格位于划分和保存之后，必须移到读取源表后、划分前，才能让新生成的路径进入输出表。源表已有完整正确路径时，也应进行核对。文件仍保留一个引用 `df_un` 的划分单元格，而其读取代码已被注释；当前二分类流程应删除或跳过该单元格，否则重启内核后顺序运行会报错。现有划分代码是按类别随机划分，尚未实现按相关任务分组。

### 2. `MobileNetV3Small.ipynb`：特征与分类

1. **加载模型**：使用 torchvision 的默认预训练 MobileNetV3 Small 权重，采用配套图像预处理。设备优先选择 Apple MPS，否则使用 CPU。首次运行需要下载权重。
2. **单张图片特征 `extract_feature()`**：读取 RGB 图像，通过 MobileNet 的特征层与全局平均池化得到 576 维特征；使用推理模式，不更新预训练网络参数。
3. **事件特征 `build_event_feature()`**：读取前三帧与后三帧；分别计算平均特征，取其逐维绝对差（576 维），再计算前后三帧共九对余弦相似度的中位数（1 维），拼接成 **577 维**事件特征。
4. **批量处理 `create_features()`**：逐行提取特征，返回特征矩阵和对应的有效数据行。遇到缺图等异常时打印 ID 与错误并跳过；当前不会自动为这些样本生成拒判结果。若没有任何有效样本，`np.stack()` 会报错。
5. **训练分类器**：使用 `LogisticRegression(max_iter=1000, class_weight="balanced")`，从有效行读取标签并训练 Pass / Fail 二分类器。
6. **预测与拒判 `final_decision()`**：获得类别概率；最高概率低于 `0.7`，或前两类概率差小于 `0.2` 时输出 `2`，否则返回概率最高的真实类别标签。
7. **整理结果**：生成内存中的 `results` 表，包含视频 ID、真实标签、最终预测、各类概率、最高概率、概率差与拒判原因。
8. **初步评价**：展示包含拒判的标签准确率、拒判数量、预测类别计数、交叉表，以及真实 Fail 的详细预测记录。

当前 notebook 不负责自动从视频取帧或检测、裁剪目标框；模型输入取决于已有图片内容。它也尚未实现模型导出、特征缓存及结果表自动保存。

## 运行方法

所需 Python 包：

```bash
pip install torch torchvision pandas numpy pillow scikit-learn openpyxl jupyterlab
```

从项目的 `code/` 目录启动 Jupyter，以匹配 notebook 中的相对路径：

```bash
cd code
jupyter lab
```

1. 先处理上文列出的 `train_test.ipynb` 执行顺序问题。
2. 确认源表每个 ID 唯一、标签一致，且图片路径与本行 ID 对应。
3. 重启内核，从头运行 `train_test.ipynb`，生成训练与测试清单。
4. 打开 `MobileNetV3Small.ipynb`，重启内核后从头运行。
5. 检查跳过提示、有效类别数量与特征维度，再查看预测结果。缺图导致的跳过会改变实际评价样本数，应单独记录。

## 当前范围与后续工作

当前是单场景、Fail 样本较少的探索性 baseline。类别平衡权重不能替代新增样本；已判定样本的准确率也不能代表全部样本或 Fail 检出能力。

后续工作包括：核对纳入与排除记录、检查相关任务的数据分组、在 development 内选择阈值、固定配置后评价测试集，并报告 Fail Precision / Recall / F1、Pass 误报率、判定覆盖率，以及画面质量、流畅度和误判案例。当前保存的 `0.7`、`0.2` 是初始阈值，不应描述为已优化或已验证的概率标准。
