# Paddle Lite · 研究分支

> 本仓库是一个以**学习与研究**为目的的 Paddle Lite 分支，不是官方发行版，也不承诺任何形式的稳定性。

本仓库 fork 自 [PaddlePaddle/Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)。在这里做的事情只有一件：读推理框架的源码，提出假设，用尽量小的改动去验证假设，再用可复现的实验说明改动是否成立。判断标准是"能不能把问题解释清楚"，性能与体积排在后面。

## 在研究什么

### 一、图优化（MIR）

- **收敛式的 pass 编排**：把一组 pass 反复执行到图不再变化为止；旧式 pass 只在第一轮跑，避免重复匹配已经被删除的节点而崩溃。默认关闭，编译时定义 `LITE_WITH_PASS_MANAGER_V2` 才启用。
- **统一的激活融合**：把散落在多个 fuser 里的激活判断收拢到一处，让"能不能融合"和"怎么融合"分开。
- **消元与折叠**：assign 消除、动态形状还原、`reshape2`+`cast` 折叠、identity transpose 消除、layer norm 折叠，以及 gelu、div-mul、scale-clip-mul 三类模式融合。
- **可观测性**：`opt` 增加 `--discarded_passes`，可以逐条关掉 pass 做对照实验。研究单个 pass 的贡献时，比改代码重新编译快得多。

### 二、ARM 内核

- 卷积与激活的融合路径，包含「无法用位标志表达的激活」如何退回通用实现。
- `sgemv` 家族的边界条件：内联汇编里的指针推进、尾部余数处理、多线程下的缓冲划分。
- 关键数学函数入口的参数校验，让越界从"随机崩溃"变成"可读的错误"。

### 三、构建与工具链

- Apple Silicon 与 Android 交叉编译的可用性修复。
- 统一 ccache、明确 cmake 版本下限、修正 SDK 与编译器探测。

## 实验方法

本分支的改动尽量满足三条：

1. **可复现**：参数、输入、期望结果写清楚，换一台机器能重跑。
2. **可归因**：改动前后用同一份输入、同一套度量各跑一遍，给出数字而不是印象。
3. **可回退**：没把握的改动用开关隔离，默认行为保持不变。

## 目录结构

```
lite/
  api/            对外接口、opt 工具
  core/           图结构、算子、MIR 优化 pass
  kernels/        各后端算子实现（arm / x86 / opencl / metal / xpu…）
  backends/       各后端底层数学库与上下文
  tools/          构建脚本、代码生成工具、融合规则扫描（fusion_rules.py）
  tests/          单元测试与模型测试
```

## 构建

cmake 用 3.22 这一档：`CMakeLists.txt` 写的下限是 3.10，但本仓记录里 3.10.3 构建不过，4.x 也没验证。编译器需要支持 C++11。常用入口：

```bash
# Android（armv8，按需加 --with_opencl=ON / --with_log=ON）
./lite/tools/build_android.sh --arch=armv8

# macOS（Apple Silicon）
./lite/tools/build_macos.sh arm64

# 模型优化工具 opt
./lite/tools/build_macos.sh build_optimize_tool

# 单元测试
./lite/tools/build_macos.sh --with_testing=ON arm64
```

产物落在 `build.lite.<平台>.<架构>[.<变体>]/` 下。

## 跑测试

```bash
# 单个 gtest（macOS 测试构建的产物，路径与源码目录对应）
./build.macos.armmacos.armv8.test/lite/core/optimizer/mir/fusion/test_conv_activation_fuse_pass

# 模型级推理
./build.macos.armmacos.armv8/lite/api/test_model_bin \
  --model_dir=<nb 目录> --use_optimize_nb=true --backend=arm_cpu \
  --input_shape=1,3,224,224
```

## 与上游的关系

- 上游：https://github.com/PaddlePaddle/Paddle-Lite
- 许可证：Apache-2.0，见 [LICENSE](LICENSE)
- 这里的实验性改动不保证合回上游，也不保证接口稳定；上游文档与正式发布以上游仓库为准。

## 免责声明

研究用代码。可能随时重构，可能包含未验证的结论和半成品开关。请勿用于对稳定性有要求的场景。
