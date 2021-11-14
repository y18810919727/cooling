# DFA ODEs for periodic cooling system

- 数据集相关介绍
https://thoughts.teambition.com/share/60d0021ab2a20c0046476e72#title=服务器集群数据集
- 状态机定义
https://thoughts.teambition.com/share/60eef9a260dba40046e9ec2d#title=状态机转换

  
### cooling制冷系统预测
- 训练脚本
```bash
cd scripts
screen -S session_name ./train_gru_dfa_sta_ns.sh
screen -S session_name ./train_gru_dfa_sta_ns.sh
```
### 温度优化实验
```bash
CUDA_VISIBLE_DEVICES=4 python optimization.py --save_dir path
```
### 根据保存的数据画图
1. sava是数据所在根目录
2. dev 处理验证集数据画图
3. test 测试集画图
4. compare 画不同模型的图（  One Neural ODE, DFA with ODE-RNN cells, DFA with H-ODE cells）
```bash
cd scripts
python readpkl.py --save path   --test  --dev --compare
```
### 画箱型图（根据不同状态）
```bash
cd getDistribution
python main.py 
```
