# TyStudy

## 介绍
Java世界的前沿AI技术的研究与学习，会放一些示例在这里，供大家一起研讨。希望借此机会，大家一起弥补Java在AI领域的缺失生态位。

- 机器学习、深度学习、神经网络等相关Demo
- AI相关的测试Case
- DJL Test Case
- Paddle FastDeploy Test Case

## BoofCV

**BoofCV** 是一个纯 Java 的全新实时计算机视觉库，其易于使用且具有非常高的性能。功能涵盖一系列Subjects、low-level图像处理、相机校准、特征检测/跟踪、运动结构、基准检测和识别。

[https://boofcv.org/](https://boofcv.org)

## DJL (Deep Java Library)

`DJL (Deep Java Library)` 是一个用于深度学习的开源、高级、与引擎无关的 Java 框架。

亚马逊（Amazon）在2019 re:Invent 上宣布推出Deep Java Library (DJL)。通过DJL，用户可以使用Java来训练和部署深度学习模型。

## FastDeploy

FastDeploy面向AI模型产业落地，是一款全场景、易用灵活、极致高效的AI推理部署工具，支持云边端部署。支持160+开箱即用的AI模型在8大类常见硬件上的部署能力。套件包括目标检测、字符识别（OCR）、人脸、人像扣图、多目标跟踪系统、NLP、Stable Diffusion文图生成、TTS 等几十种任务场景。

## DJL Paddle FastDeploy 示例

百度飞浆 `FastDeploy` 产品的示例，基于`DJL Python Engine`实现。

**准备工作：**

- 请参考百度飞浆`FastDeploy`官网（[https://www.paddlepaddle.org.cn/fastdeploy](https://www.paddlepaddle.org.cn/fastdeploy)），安装FastDeploy；

- 配置环境变量 **PYTHON_EXECUTABLE**  （Python执行文件的完整路径）

  > 如：PYTHON_EXECUTABLE = C:\DevTools\miniconda\envs\cuda117\python.exe

  > 注：Python环境建议用 **Anoconda** 或 **miniConda** 搭建。

- Clone 本项目，运行 [src/main/java/com/ty/test/ai/cv/PPYoloEPlusModelTest.java](https://gitee.com/tommycloud/TyStudy/blob/master/src/main/java/com/ty/test/ai/cv/PPYoloEPlusModelTest.java) 查看效果。

```sh
# 成功运行输出如下：
2024-05-03 22:21:26.493 [W-7380-ppyoloEplus-stdout] INFO  ai.djl.python.engine.PyProcess - W-7380-ppyoloEplus-stdout: [INFO] fastdeploy/runtime/runtime.cc(273)::fastdeploy::Runtime::CreatePaddleBackend	Runtime initialized with Backend::PDINFER in Device::GPU.
2024-05-03 22:21:26.500 [W-7380-ppyoloEplus-stdout] INFO  ai.djl.python.engine.PyProcess - W-7380-ppyoloEplus-stdout: PPYoloE Plus Model Loaded, spend time 6 second.
2024-05-03 22:21:27.050 [main] INFO  c.ty.test.ai.cv.PPYoloEPlusModelTest - 推理结束：[
	{"class": "风筝", "probability": 0.97197, "bounds": {"x"=161.763, "y"=81.466, "width"=455.528, "height"=263.177}}
	{"class": "人", "probability": 0.90937, "bounds": {"x"=163.771, "y"=81.891, "width"=34.823, "height"=85.555}}
	{"class": "人", "probability": 0.90780, "bounds": {"x"=267.230, "y"=89.545, "width"=31.428, "height"=79.574}}
	{"class": "人", "probability": 0.89662, "bounds": {"x"=581.122, "y"=113.050, "width"=31.693, "height"=85.292}}
	{"class": "人", "probability": 0.88805, "bounds": {"x"=414.957, "y"=89.496, "width"=92.144, "height"=195.913}}
	{"class": "人", "probability": 0.80455, "bounds": {"x"=363.936, "y"=58.531, "width"=17.751, "height"=55.904}}
	{"class": "人", "probability": 0.79903, "bounds": {"x"=503.756, "y"=113.386, "width"=91.321, "height"=157.999}}
	{"class": "椅子", "probability": 0.79050, "bounds": {"x"=75.447, "y"=121.782, "width"=28.761, "height"=34.099}}
	{"class": "人", "probability": 0.78817, "bounds": {"x"=104.796, "y"=45.746, "width"=22.949, "height"=47.837}}
	{"class": "人", "probability": 0.76899, "bounds": {"x"=327.770, "y"=39.126, "width"=19.448, "height"=41.508}}
	{"class": "人", "probability": 0.72002, "bounds": {"x"=348.658, "y"=44.040, "width"=19.439, "height"=54.283}}
	{"class": "人", "probability": 0.69320, "bounds": {"x"=378.745, "y"=40.857, "width"=16.668, "height"=43.075}}
	{"class": "人", "probability": 0.62536, "bounds": {"x"=26.247, "y"=118.027, "width"=35.093, "height"=35.516}}
	{"class": "椅子", "probability": 0.61076, "bounds": {"x"=98.472, "y"=130.829, "width"=18.842, "height"=25.674}}
	{"class": "人", "probability": 0.52820, "bounds": {"x"=169.729, "y"=47.327, "width"=8.937, "height"=13.904}}
	{"class": "背包", "probability": 0.50318, "bounds": {"x"=65.103, "y"=134.649, "width"=19.887, "height"=19.887}}
]
```

![test1.jpg预测结果](https://gitee.com/tommycloud/TyStudy/raw/master/src/main/resources/test/test1_visualized_result.jpg)

## DJL Paddle ONNX 示例

使用 `Paddle2onnx`工具将Paddle模型转换为ONNX模型，基于`DJL ONNX Engine`实现 AI 推理。

- Clone 本项目，运行 [src/main/java/com/ty/test/ai/cv/PPYoloEPlusONNXModelTest.java](https://gitee.com/tommycloud/TyStudy/blob/master/src/main/java/com/ty/test/ai/cv/PPYoloEPlusONNXModelTest.java) 查看效果。

```sh
# 成功运行输出如下：
2024-05-15 00:24:33.943 [main] DEBUG ai.djl.engine.Engine - Found default engine: PyTorch
2024-05-15 00:24:37.986 [main] DEBUG ai.djl.repository.zoo.ModelZoo - Checking ModelLoader: ai.djl.localmodelzoo:ppyoloE_plus UNDEFINED [
ai.djl.localmodelzoo/ppyoloE_plus/ppyoloE_plus {}
]
2024-05-15 00:24:38.001 [main] DEBUG ai.djl.repository.MRL - Preparing artifact: file:/opt/onnx_models/ppyoloE_plus/, ai.djl.localmodelzoo/ppyoloE_plus/ppyoloE_plus {}
2024-05-15 00:24:38.002 [main] DEBUG ai.djl.repository.SimpleRepository - Skip prepare for local repository.
Loading:     100% |========================================|
2024-05-15 00:24:40.924 [main] INFO  c.t.a.c.p.m.p.PPYoloEPlusONNXModel - PPYoloE Plus ONNX模型加载完毕
2024-05-15 00:24:40.925 [main] INFO  c.t.t.ai.cv.PPYoloEPlusONNXModelTest - 开始推理：src\main\resources\test\test1.jpg
2024-05-15 00:24:40.925 [main] DEBUG c.t.a.c.p.t.PaddleDetectionTranslator - Process Input 数据预处理...
2024-05-15 00:24:40.991 [main] DEBUG c.t.a.c.p.t.PaddleDetectionTranslator - Process Input 数据预处理完成，耗时：66ms
2024-05-15 00:24:41.695 [main] DEBUG c.t.a.c.p.t.PaddleDetectionTranslator - Process Output 数据后处理...
2024-05-15 00:24:41.704 [main] DEBUG c.t.a.c.p.t.PaddleDetectionTranslator - Process Output 数据后处理完成，耗时：8ms
2024-05-15 00:24:42.259 [main] TRACE c.t.a.c.p.t.PaddleDetectionTranslator - 
目标检测结果集（300条）                                                         
label  |         score          |         x_min         |        y_min         |        x_max         |        y_max         |
----------------------------------------------------------------------------------------------------------------------------
33  |    0.9723131060600281  |   161.77688598632812  |   81.42156982421875  |    616.650146484375  |   344.8529052734375  |
0  |    0.9130071401596069  |    163.8280487060547  |   81.87583923339844  |   198.5448455810547  |   167.5631103515625  |
0  |      0.90949547290802  |   267.20965576171875  |    89.6672592163086  |    298.622314453125  |  169.03573608398438 
...  |                   ...  |                  ...  |                 ...  |                 ...  |                 ... 
2024-05-15 00:24:42.329 [main] INFO  c.t.t.ai.cv.PPYoloEPlusONNXModelTest - 推理结束 src\main\resources\test\test1.jpg 	目标检测数量：15   
```

![test1.jpg预测结果](https://gitee.com/tommycloud/TyStudy/raw/master/src/main/resources/test/test1_visualized_result.jpg)

## 使用的开源算法

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

## 预训练模型

- PPYoloE Plus 预训练模型下载地址

  > https://pan.baidu.com/s/1EFbAoHujknZgrtlzsA4t0Q?pwd=a42d
  >
  
- PPYoloE Plus ONNX 预训练模型下载地址

  > https://pan.baidu.com/s/1aK1vNSd-I1l7TB_e0_Df6w?pwd=sdty