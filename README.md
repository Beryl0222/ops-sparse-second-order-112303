# 有效支撑可变化的稀疏二阶算子

`data/record.json` 是 有效支撑可变化的稀疏二阶算子 使用的本地合成交换样本。数组按 JSON 的下标顺序排列。本文的坐标、单位和枚举约定与该记录配套。记录中如果有 `scale` 或 `limits`，对应同一输入协议的部署规格。

视图声明为 [参数名, 原索引列表]。表达式引用 [视图名, 该视图内下标]，affine 的每项是 [系数, 引用]，最后一项是偏置。product 表示乘积，positive_part 表示 max(0,x)。总能量是启用项的 weight × residual² 之和，带符号权重允许同一相互作用抵消。状态 edge 按十进制值位于门槛，允许 1e-12 的边界辨认容差。directions 各分量及 coordinates 的行列下标按 requests.vars 排列，其他变量固定。混合二阶项按其列坐标方向上对应的梯度分量变化定义，方向积按指定方向上的梯度变化定义。

## 读取样本

使用 Python 3.11 或 3.12，并在虚拟环境安装锁定依赖。读取入口返回原始记录。作为命令运行时，它会列出各顶层输入项目以及 NumPy 版本。

```sh
python -m pip install -r requirements.txt
python -m sparse_second_order_input
python -m unittest discover -s tests -v
```

测试读取交换样本，核对其中的张量尺寸、标识与数据约定。

## 稀疏二阶算子

`sparse_second_order.py` 在同一份交换契约上构建可微能量图，按求值实例为优化器与诊断端交付二阶结果，全程不物化整张 Hessian。

- `Session.build_graph(record)` 解析参数、共享视图、带惩罚权重的残差与部署规格（`scale` 或 `limits` 中的 `scratch_bytes` 作为工作预算）。换一张图即再构建一次。
- `graph.evaluation(state=…, request=…)` 绑定求值点（状态覆盖或 `point=` 覆盖）、启用掩码与请求变量范围，返回求值实例；仅有效分量进入该次目标。内容相同的调用返回同一实例，可用 `inst.token` 经 `session.instance(token)` 按当时的求值实例查回旧结果。
- `inst.subblock(coords)` 以稀疏坐标读回指定子块；`inst.direction_products(dirs)` 批量乘若干方向。坐标与方向分量按 `requests.vars` 排列，其他变量参与求值但固定；输出对原变量编号，并尊重同一原变量经多视图进入多个残差的别名。
- 每个条目区分三种情形：`determined` 真零（数值抵消但 `structural=True` 保留交互可能出现；结构上无通路则 `structural=False`）、`determined` 非零、`undetermined`（当前图在该交叉项不可微，附受影响变量作用域与两侧单侧极限，不默认采用某侧二阶导数）。远离作用域的已确定项与方向积照常交付。
- 软阈 `positive_part` 以实际求值点判定开闭（|门值| ≤ 1e-12 视为门槛）。方向积逐分量判定：零方向或从两边接近都留在门槛上的方向确定可用；跨过门槛且两侧不一致的分量标出。优化器可据 `DirectionResult.usable` 与 `curvature` 决定是否采用更新。
- 块结构证据按（请求变量、启用掩码、软阈分支）签名缓存复用，同一图的连续求值点不重新发现结构；掩码或阈值跨越触发重新发现。数值结果按实例缓存，新出现的交叉项不沿旧试探身份相加。
- 计算按需求只触达可达残差（静态倒排 + 逐残差局部求值），超出工作预算时抛出 `BudgetExceededError`。

```sh
python -m sparse_second_order   # 打印样本三个状态的子块、方向积与作用域
```
