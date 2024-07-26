## AudioMatic

AudioMatic是一个用于预处理音频的项目，支持Python >= 3.8。

### 环境搭建

按照以下步骤进行环境搭建：

#### 克隆项目仓库

```sh
git clone https://github.com/AssassinQuin/AudioMatic.git

cd AudioMatic
```

#### 创建并激活 Conda 环境

```sh
conda create --name audioMatic python=3.8 -y

conda activate audioMatic
```

#### 安装项目依赖

```sh
pip install -r requirements.txt
```

### 下载模型

#### uvr5模型下载:[huggingface](https://huggingface.co/Delik/uvr5_weights/tree/main) | [modelscope](https://modelscope.cn/models/AI-ModelScope/uvr5_weights)

```sh
# huggingface 下载:
git lfs install
git clone https://huggingface.co/Delik/uvr5_weights
mv uvr5_weights /path/AudioMatic/uvr5

# modelscope 下载:
git clone https://www.modelscope.cn/AI-ModelScope/uvr5_weights.git
mv uvr5_weights /path/AudioMatic/uvr5
```

### 修改流程

根据需要在 `main.py`修改流程：

```python
def initialize_strategies(root_path, model_weights_root, timestamp, device):
    """
    流程：
    1. 将音频文件转换为wav格式
    2. 切割音频文件，把长文件切割为较短文件，若显存不足/较多 可以酌情切割
    3. 提取人声
    4. 分类说话人
    """
    # 将音频文件转换为wav格式策略
    convert_to_wav_strategy = ConvertToWavStrategy(root_path, timestamp)
    # 合并wav策略
    merge_audio_strategy = MergeAudioStrategy(root_path, timestamp)
    # 切割音频文件策略（参数自行调整）
    cut_strategy = CutAudioStrategy(root_path, timestamp, device)
    # 提取人声策略
    extract_vocal_strategy = ExtractVocalStrategy(
        root_path, model_weights_root, timestamp, device
    )
    # 分类说话人策略
    classify_strategy = ClassifyAudioStrategy(root_path, timestamp, device)
    # 返回顺序是执行顺序
    return [
        convert_to_wav_strategy,
        merge_audio_strategy,
        cut_strategy,
        extract_vocal_strategy,
        merge_audio_strategy,
        classify_strategy,
    ]

```

### 贡献

欢迎对本项目进行贡献！请提交Pull Request或报告问题。

### 许可证

该项目使用 MIT 许可证。详情请参阅 LICENSE 文件。