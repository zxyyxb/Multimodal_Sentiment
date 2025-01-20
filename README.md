## 本项目是《人工智能》的第五次实验，多模态情感分析

## 安装依赖
进入该项目文件夹后终端输入 pip install -r requirements.txt

├── src  #包含训练数据和预测结果 
│   ├── data #数据
│   │   ├── 1.jpg
│   │   ├── 1.txt
│   │   ├── 2.jpg
│   │   ├── 2.txt
│   │   └── ......
│   ├── test_predicted_text_only.txt  #只用txt训练的结果
│   ├── test_predicted.txt   #用image和txt训练的结果
│   └── test_without_label.txt  #测试集
│   └── train.txt  #训练集txt
├── img_only.py   #只用img训练
├── txt_only.py   #只用txt训练
├── README.md   
├── requirements.txt  #依赖安装
├── main.py  #用image和txt训练
├── 实验报告.pdf


## 运行说明
安装好依赖以后运行main.py（已经包含数据处理，模型构建，训练，结果输出等）或者img_only.py或者txt_only.py即可得到相应的预测结果

## 参考的库
此代码的某些部分基于以下库：
- [PyTroch](https://github.com/pytorch/pytorch)
- [Transformers (by Hugging Face)](https://github.com/huggingface/transformers)
- [Pandas](https://github.com/pandas-dev/pandas)
- [torchivision](https://github.com/pytorch/vision)
