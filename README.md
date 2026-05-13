# Assess_Vision_27

Deus 战队 RoboMaster 2027 赛季视觉组招新考核模板仓库。

## 用途

本仓库为阶段二考核提供 C++ 工程模板，考核内容为：基于阶段一训练导出的 ONNX 模型，在 C++ 中完成装甲板检测推理并输出中心点坐标。

仓库内置 Judge 日志记录和 Evaluate 自动评分模块，用于验收时量化评估精度与速度。

## 结构

```
detect/
├── CMakeLists.txt          # 构建配置
├── README.md               # 阶段二详细说明
├── include/                # 头文件
├── src/                    # 源文件 (需要填充实现)
├── assets/                 # 测试素材与 ground truth
└── results/                # 运行结果
```

具体任务说明和评估标准见 `detect/README.md`。
