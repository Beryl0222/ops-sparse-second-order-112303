# 有效支撑可变化的稀疏二阶算子

`data/record.json` 是 有效支撑可变化的稀疏二阶算子 使用的本地合成交换样本。数组按 JSON 的下标顺序排列。本文的坐标、单位和枚举约定与该记录配套。记录中如果有 `scale` 或 `limits`，对应同一输入协议的部署规格。

视图声明为 [参数名, 原索引列表]。表达式引用 [视图名, 该视图内下标]，affine 的每项是 [系数, 引用]，最后一项是偏置。product 表示乘积，positive_part 表示 max(0,x)。能量定义为启用残差的 weight × residual² 之和，负系数用于合成抵消检验。状态标识 edge 的输入根据十进制原值正好位于门槛，计算中允许 1e-12 的边界辨认容差。子块二阶项仅在其导数依赖不可微的表达式时不可用，方向导数为零或能证明不会受该分支切换影响时仍可提供。

## 读取样本

使用 Python 3.11 或 3.12，并在虚拟环境安装锁定依赖。读取入口返回原始记录。作为命令运行时，它会列出各顶层输入项目以及 NumPy 版本。

```sh
python -m pip install -r requirements.txt
python -m sparse_second_order_input
python -m unittest discover -s tests -v
```

测试读取交换样本，核对其中的张量尺寸、标识与数据约定。
