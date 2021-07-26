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

### 制冷系统功率分析

```bash
CUDA_VISIBLE_DEVICES=4 python optimization.py
```
