# AVSC
### 环境配置
并没有用到额外的包，只需要将该任务原本配好的环境重命名为common就行

或者运行如下脚本
```bash
conda env create -f environment.yml
conda activate common
```
### 特征提取
使用 openl3 提取的 audio 和 visual 特征，此步骤过程较长，已预先提取好放于 `/dssg/home/acct-stu/stu464/ai3611/av_scene_classify/data/feature`，基于进行实验

### 实验运行
#### Mean Model
所有的Mean Model(一开始在时间轴上进行均值操作)都在./mean_model中，可以通过以下脚本运行
```angular2html
conda activate common
python train_mean.py --config_file configs/name.yaml --cuda 0

# evaluation
python evaluate.py --experiment_path experiments/name
```
其中name为模型名字
#### 复现最优性能
有2个模型都能达到最优性能

1. Mid Fusion的调参版
```angular2html
conda activate common
python train_mean.py --config_file configs/baseline.yaml --cuda 0

# evaluation
python evaluate.py --experiment_path experiments/mid
```

2. 划窗改进
```angular2html
conda activate common
python train_mean.py --config_file configs/com_dt_t.yaml --cuda 0

# evaluation
python evaluate.py --experiment_path experiments/com_dt_t
```
#### Conv Model
所有的Mean Model(卷积模型)都在./conv_model中，可以通过以下脚本运行
```angular2html
conda activate common
python train_conv.py --config_file configs/name.yaml --cuda 0

# evaluation
python evaluate.py --experiment_path experiments/name
```
其中name为模型名字，运行时需要把evaluate.py中的
```angular2html
from mean_model import load_model
```
改为
```angular2html
from conv_model import load_model
```
#### Mean Model中各模型名字解释
所有的Mean Model都在./mean_model中，通过__init__.py中定义的load_model函数调用。其中：

1. audio.py 指 Audio Only
2. video.py 指 Video Only
3. early.py 指 Early Fusion
4. mid.py 指 Mid Fusion，也即为Baseeline模型
5. decision.py 指 Decision Level Fusion
6. audio_vattn.py 指 Audio + Vedio Attention
7. video_aattn.py 指 Video + Audio Attention
8. video_divide_t.py 指 Video Only + 划窗操作
9. audio_divide_t.py 指 Audio Only + 划窗操作
10. com_dt_t.py 指 Baseline + 划窗操作
11. decision_midattn 指 Decision Level Fusion + AV Attention


