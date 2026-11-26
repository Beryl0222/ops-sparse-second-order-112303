"""有效支撑可变化的稀疏二阶算子。

在交换契约给出的可微能量图上，按求值实例为优化器与诊断端交付稀疏二阶结果：
指定子块的稀疏坐标读回、批量方向积，以及软阈门槛处的双侧判定与作用域报告。

设计约定
--------
- 不物化整张 Hessian：按请求变量经静态倒排找到可达残差，逐残差局部求值，
  只累加请求坐标与请求方向需要的结果；不可达残差对能量只做流式求值，不进缓存。
- 结构证据（可能出现的交互对、变量邻接、可达残差）按签名缓存复用。签名包含
  请求变量、启用掩码与每个可达含软阈残差的分支；掩码变化或阈值跨越会得到新
  签名，从而重新发现结构。数值相互抵消不会收缩结构证据。
- 数值结果（子块条目、方向积）按求值实例缓存；不同实例之间绝不沿旧试探身份
  累加新出现的交叉项。
- 软阈（positive_part）位于门槛时，该残差二阶不可微：行、列（或方向）同时落在
  门作用域内的条目标为 undetermined，并给出两侧单侧极限；作用域之外的已确定项
  照常交付。零方向或使门值保持不动的方向按同一口径判定为确定。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, NamedTuple

DEFAULT_BOUNDARY_TOL = 1e-12  # 门槛辨认容差（见 README）
_ENTRY_BYTES = 64             # 预算估计：单条稀疏条目的字节开销
_EVAL_BASE_BYTES = 128        # 预算估计：单次残差求值缓存的基础开销
_PRODUCT_BASE_BYTES = 128     # 预算估计：单条方向积的基础开销

__all__ = [
    "Budget",
    "BudgetExceededError",
    "ContractError",
    "EnergyGraph",
    "Evaluation",
    "Session",
    "Var",
    "DEFAULT_BOUNDARY_TOL",
]

_OVERRIDE_RE = re.compile(r"^(.+)_index(\d+)$")


class ContractError(ValueError):
    """交换契约或请求不合法。"""


class BudgetExceededError(RuntimeError):
    """调用会超出工作预算；已按需求截断，未物化更多结果。"""


class Var(NamedTuple):
    """原变量编号：参数名 + 物理变量号。"""

    param: str
    index: int

    def as_list(self):
        return [self.param, self.index]


def _as_var(item) -> Var:
    if isinstance(item, Var):
        return item
    if isinstance(item, (list, tuple)) and len(item) == 2:
        param, index = item
        if isinstance(param, str) and isinstance(index, int):
            return Var(param, index)
    raise ContractError(f"无法识别的变量引用：{item!r}")


def _canon(u: Var, v: Var):
    return (u, v) if u <= v else (v, u)


# ---------------------------------------------------------------------------
# 表达式图
# ---------------------------------------------------------------------------


class Expr:
    """残差表达式节点基类。"""

    __slots__ = ()


@dataclass(frozen=True)
class Ref(Expr):
    """视图引用，建图时已解析为原变量。"""

    var: Var


@dataclass(frozen=True)
class Const(Expr):
    value: float


@dataclass(frozen=True)
class Affine(Expr):
    """仿射映射：terms 为 (系数, 引用) 序列，bias 为偏置。"""

    terms: tuple
    bias: float


@dataclass(frozen=True)
class Product(Expr):
    """成对乘积。"""

    left: Expr
    right: Expr


@dataclass(frozen=True)
class PositivePart(Expr):
    """软阈项 max(0, x)。"""

    arg: Expr


def parse_expr(node, views: Mapping[str, tuple]) -> Expr:
    """把契约 JSON 表达式解析为表达式树，视图引用解析为原变量。"""
    if isinstance(node, (int, float)):
        return Const(float(node))
    if isinstance(node, (list, tuple)) and node:
        if len(node) == 2 and isinstance(node[0], str) and isinstance(node[1], int):
            view = views.get(node[0])
            if view is None:
                raise ContractError(f"未知视图：{node[0]!r}")
            if not 0 <= node[1] < len(view):
                raise ContractError(f"视图 {node[0]!r} 下标越界：{node[1]}")
            return Ref(view[node[1]])
        op = node[0]
        if op == "product" and len(node) == 3:
            return Product(parse_expr(node[1], views), parse_expr(node[2], views))
        if op == "positive_part" and len(node) == 2:
            return PositivePart(parse_expr(node[1], views))
        if op == "affine" and len(node) == 3:
            terms = []
            for item in node[1]:
                coef, ref = item
                ref = parse_expr(ref, views)
                if not isinstance(ref, Ref):
                    raise ContractError("affine 的项应为 [系数, 引用]")
                terms.append((float(coef), ref))
            return Affine(tuple(terms), float(node[2]))
    raise ContractError(f"未知算子或残缺表达式：{node!r}")


def _support(expr: Expr) -> frozenset:
    """表达式静态支撑（引用的原变量集合），与求值点无关。"""
    if isinstance(expr, Ref):
        return frozenset({expr.var})
    if isinstance(expr, Const):
        return frozenset()
    if isinstance(expr, Affine):
        return frozenset(ref.var for _, ref in expr.terms)
    if isinstance(expr, Product):
        return _support(expr.left) | _support(expr.right)
    if isinstance(expr, PositivePart):
        return _support(expr.arg)
    raise ContractError(f"未知表达式节点：{expr!r}")


def _has_gate(expr: Expr) -> bool:
    if isinstance(expr, PositivePart):
        return True
    if isinstance(expr, Product):
        return _has_gate(expr.left) or _has_gate(expr.right)
    return False


# ---------------------------------------------------------------------------
# 局部求值
# ---------------------------------------------------------------------------


@dataclass
class GateInfo:
    """一次软阈门槛辨认：门值在容差内，记录门参数的偏导（作用域）。"""

    partials: dict          # Var -> 门参数对该变量的偏导（已剔除零项）
    gate_value: float       # 辨认时的门值（|门值| <= 容差）
    node: PositivePart      # 产生该信息的表达式节点（识别顶层门）
    top_level: bool = False


@dataclass
class ResidualEval:
    """单个残差在某求值点的局部结果。"""

    value: float
    grad: dict              # Var -> ∂ρ/∂θ（稀疏）
    hess: dict              # (Var, Var) 规范键 -> ∂²ρ（稀疏）
    gates: tuple            # 门槛辨认信息（空表示可微）
    silent: bool            # 因软阈关闭而在邻域内恒零


def _eval_expr(expr: Expr, point: Mapping[str, tuple], tol: float):
    """返回 (值, 梯度, 二阶稀疏表, 门槛信息列表, 是否恒零)。"""
    if isinstance(expr, Ref):
        var = expr.var
        return point[var.param][var.index], {var: 1.0}, {}, [], False
    if isinstance(expr, Const):
        return expr.value, {}, {}, [], False
    if isinstance(expr, Affine):
        value = expr.bias
        grad = {}
        for coef, ref in expr.terms:
            var = ref.var
            value += coef * point[var.param][var.index]
            grad[var] = grad.get(var, 0.0) + coef
        return value, grad, {}, [], False
    if isinstance(expr, Product):
        va, ga, ha, ca, sa = _eval_expr(expr.left, point, tol)
        vb, gb, hb, cb, sb = _eval_expr(expr.right, point, tol)
        grad = {}
        for u, du in ga.items():
            grad[u] = grad.get(u, 0.0) + vb * du
        for u, du in gb.items():
            grad[u] = grad.get(u, 0.0) + va * du
        hess = {}
        for key, m in ha.items():
            hess[key] = hess.get(key, 0.0) + vb * m
        for key, m in hb.items():
            hess[key] = hess.get(key, 0.0) + va * m
        # ∇²(ab) 的交叉部分 a_u·b_v + a_v·b_u：按规范键累加，对角加倍
        for u, au in ga.items():
            for v, bv in gb.items():
                key = _canon(u, v)
                hess[key] = hess.get(key, 0.0) + au * bv * (2.0 if u == v else 1.0)
        return va * vb, grad, hess, ca + cb, sa or sb
    if isinstance(expr, PositivePart):
        value, grad, hess, gates, silent = _eval_expr(expr.arg, point, tol)
        if silent:
            # 门参数在邻域内恒零，本项同样恒零
            return 0.0, {}, {}, gates, True
        if value > tol:
            return value, grad, hess, gates, False
        if value < -tol:
            return 0.0, {}, {}, gates, True
        # 门槛上：能量梯度两侧一致（取 0），二阶不可微，记录作用域
        info = GateInfo(dict(grad), value, expr)
        return 0.0, {}, {}, gates + [info], False
    raise ContractError(f"未知表达式节点：{expr!r}")


def _eval_residual_expr(residual: "Residual", point: Mapping[str, tuple], tol: float) -> ResidualEval:
    value, grad, hess, gates, silent = _eval_expr(residual.expr, point, tol)
    cleaned = []
    for info in gates:
        info.partials = {u: du for u, du in info.partials.items() if du != 0.0}
        if not info.partials:
            continue  # 门参数梯度为零，两侧一致，无需标记
        info.top_level = info.node is residual.expr
        cleaned.append(info)
    return ResidualEval(value, grad, hess, tuple(cleaned), silent)


def _branch_of(ev: ResidualEval) -> str:
    if ev.silent:
        return "inactive"
    if ev.gates:
        return "threshold"
    return "active"


@dataclass(frozen=True)
class Residual:
    """带惩罚权重的残差分量。"""

    rid: str
    expr: Expr
    weight: float
    loss: str
    support: frozenset      # 静态支撑（原变量）
    gated: bool             # 是否含软阈


@dataclass(frozen=True)
class Budget:
    """工作预算：scratch_bytes 为估计的暂存字节上限，None 表示不限。"""

    scratch_bytes: int | None = None


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UndeterminedCause:
    """条目或分量不可定的原因：某残差在门槛处二阶不可微。"""

    residual: str
    kind: str               # "positive_part_threshold"
    scope: tuple            # 受影响变量作用域（原变量编号）
    toward: float | None    # 沿 +列方向/方向的单侧极限（嵌套门为 None）
    away: float | None      # 沿 -方向的单侧极限

    def to_dict(self):
        return {
            "residual": self.residual,
            "kind": self.kind,
            "scope": [v.as_list() for v in self.scope],
            "toward": self.toward,
            "away": self.away,
        }


@dataclass(frozen=True)
class EntryResult:
    """子块条目：按原变量编号的稀疏坐标读回。"""

    row: Var
    col: Var
    row_pos: int | None
    col_pos: int | None
    status: str             # "determined" / "undetermined"
    value: float | None     # determined 时的值（真零为 0.0）
    structural: bool        # 结构证据：该交互可能出现
    determined_part: float  # 已确定部分之和（undetermined 时仅为部分量）
    causes: tuple

    def to_dict(self):
        return {
            "row": self.row.as_list(),
            "col": self.col.as_list(),
            "request_pos": None if self.row_pos is None else [self.row_pos, self.col_pos],
            "status": self.status,
            "value": self.value,
            "structural": self.structural,
            "determined_part": self.determined_part,
            "causes": [c.to_dict() for c in self.causes],
        }


@dataclass(frozen=True)
class ComponentResult:
    """方向积的一个分量（按请求序，标注原变量）。"""

    var: Var
    pos: int
    status: str             # "determined" / "undetermined"
    value: float | None
    determined_part: float
    causes: tuple

    def to_dict(self):
        return {
            "var": self.var.as_list(),
            "pos": self.pos,
            "status": self.status,
            "value": self.value,
            "determined_part": self.determined_part,
            "causes": [c.to_dict() for c in self.causes],
        }


@dataclass(frozen=True)
class DirectionResult:
    """一次方向积：H·d 的稀疏交付。"""

    direction: tuple
    status: str             # "determined" / "partial"
    components: tuple       # ComponentResult，按请求序
    curvature: float | None  # dᵀHd；方向非零分量均可定时给出

    @property
    def usable(self) -> bool:
        """优化器口径：请求方向上全部分量确定才可采用更新。"""
        return self.status == "determined"

    def values(self):
        """各分量值组成的元组（含不可定分量的 None），按请求序。"""
        return tuple(c.value for c in self.components)

    def to_dict(self):
        return {
            "direction": list(self.direction),
            "status": self.status,
            "curvature": self.curvature,
            "components": [c.to_dict() for c in self.components],
        }


@dataclass(frozen=True)
class ScopeReport:
    """未定义二阶的分量及其受影响变量作用域。"""

    residual: str
    kind: str
    scope: tuple            # 原变量编号
    gate_value: float
    top_level: bool
    reaches_request: bool   # 作用域是否触及请求变量

    def to_dict(self):
        return {
            "residual": self.residual,
            "kind": self.kind,
            "scope": [v.as_list() for v in self.scope],
            "gate_value": self.gate_value,
            "top_level": self.top_level,
            "reaches_request": self.reaches_request,
        }


@dataclass(frozen=True)
class StructureEvidence:
    """块结构证据：某签名下可能出现的交互对、变量邻接与可达残差。"""

    signature: tuple
    pairs: frozenset        # 规范键 (Var, Var)，可能出现交互
    adjacency: Mapping      # Var -> 残差序号元组
    relevant: tuple         # 参与该次结构的残差序号


# ---------------------------------------------------------------------------
# 求值实例
# ---------------------------------------------------------------------------


class Evaluation:
    """一次求值实例：同一求值点上为优化器与诊断端交付二阶结果。"""

    def __init__(self, graph, point, enabled, variables, directions, coordinates, token):
        self._graph = graph
        self._point = {k: tuple(v) for k, v in point.items()}
        self._enabled = frozenset(enabled)
        self._enabled_ordered = tuple(r.rid for r in graph._residuals if r.rid in self._enabled)
        self._vars = tuple(variables)
        self._varset = frozenset(self._vars)
        self._pos = {v: k for k, v in enumerate(self._vars)}
        self._req_directions = directions
        self._req_coordinates = coordinates
        self.token = token
        self._eval_cache = {}
        self._payload_cache = {}
        self._entry_cache = {}
        self._block_cache = {}
        self._product_cache = {}
        self._batch_cache = {}
        self._value_cache = None
        self._grad_cache = None
        self._scopes_cache = None
        self._used_bytes = 0
        self._stats = {"residual_evals": 0, "cached_evals": 0, "structure_reused": False}
        self._prepare()

    # -- 基本属性 ---------------------------------------------------------

    @property
    def request_vars(self):
        """请求变量范围（原变量编号，请求序）。"""
        return self._vars

    @property
    def enabled(self):
        """本次启用的残差标识（按契约顺序）。"""
        return self._enabled_ordered

    @property
    def point(self):
        """求值点（参数名 -> 值列表）。"""
        return {k: list(v) for k, v in self._point.items()}

    @property
    def structure(self) -> StructureEvidence:
        return self._evidence

    @property
    def structure_signature(self):
        return self._evidence.signature

    def __repr__(self):
        return f"<Evaluation {self.token} vars={len(self._vars)}>"

    # -- 预算与统计 -------------------------------------------------------

    def _spend(self, n_bytes, what):
        self._used_bytes += n_bytes
        limit = self._graph._budget.scratch_bytes
        if limit is not None and self._used_bytes > limit:
            raise BudgetExceededError(
                f"{what}会超出工作预算：已计 {self._used_bytes} 字节，上限 {limit} 字节；"
                "请缩小请求变量范围或提高预算"
            )

    def stats(self):
        """工作统计：残差求值次数、缓存占用、结构复用等。"""
        return {
            "token": self.token,
            "residual_evals": self._stats["residual_evals"],
            "cached_evals": self._stats["cached_evals"],
            "scratch_bytes": self._used_bytes,
            "structure_reused": self._stats["structure_reused"],
            "candidates": len(self._candidates),
            "relevant": len(self._evidence.relevant),
        }

    # -- 结构准备 ---------------------------------------------------------

    def _prepare(self):
        graph = self._graph
        cand = set()
        for v in self._vars:
            cand.update(graph._by_var.get(v, ()))
        self._candidates = sorted(cand)
        branch_items = []
        for ridx in self._candidates:
            r = graph._residuals[ridx]
            if not r.gated or r.rid not in self._enabled:
                continue
            ev = self._eval_residual(ridx)
            branch_items.append((r.rid, _branch_of(ev)))
        signature = (self._vars, tuple(sorted(self._enabled)), tuple(branch_items))
        evidence = graph._structures.get(signature)
        reused = evidence is not None
        if evidence is None:
            evidence = self._discover_structure(signature)
            graph._structures[signature] = evidence
        self._evidence = evidence
        self._stats["structure_reused"] = reused

    def _discover_structure(self, signature) -> StructureEvidence:
        graph = self._graph
        adjacency = {v: [] for v in self._vars}
        pairs = set()
        relevant = []
        for ridx in self._candidates:
            r = graph._residuals[ridx]
            if r.rid not in self._enabled or r.weight == 0.0:
                continue
            ev = self._eval_residual(ridx)
            if ev.silent:
                continue  # 软阈关闭，该残差在邻域内恒零
            relevant.append(ridx)
            supp = [v for v in r.support if v in self._varset]
            for a in supp:
                adjacency[a].append(ridx)
                for b in supp:
                    if a <= b:
                        pairs.add((a, b))
        return StructureEvidence(
            signature,
            frozenset(pairs),
            {v: tuple(lst) for v, lst in adjacency.items()},
            tuple(relevant),
        )

    def structure_pairs(self):
        """该实例结构证据下可能出现的交互对（按原变量编号排序）。"""
        return tuple(sorted(self._evidence.pairs))

    # -- 残差求值 ---------------------------------------------------------

    def _eval_residual(self, ridx) -> ResidualEval:
        ev = self._eval_cache.get(ridx)
        if ev is None:
            r = self._graph._residuals[ridx]
            ev = _eval_residual_expr(r, self._point, self._graph._tol)
            self._stats["residual_evals"] += 1
            self._stats["cached_evals"] += 1
            self._eval_cache[ridx] = ev
            self._spend(
                _EVAL_BASE_BYTES + (len(ev.grad) + len(ev.hess)) * _ENTRY_BYTES,
                "残差求值缓存",
            )
        return ev

    def _eval_residual_streaming(self, ridx) -> ResidualEval:
        """不可达残差的流式求值：只累计能量，不进缓存。"""
        ev = self._eval_cache.get(ridx)
        if ev is not None:
            return ev
        self._stats["residual_evals"] += 1
        return _eval_residual_expr(
            self._graph._residuals[ridx], self._point, self._graph._tol
        )

    # -- 一阶量 -----------------------------------------------------------

    def value(self) -> float:
        """本次目标：启用分量的 weight × residual² 之和。"""
        if self._value_cache is None:
            total = 0.0
            for ridx, r in enumerate(self._graph._residuals):
                if r.rid not in self._enabled:
                    continue
                ev = self._eval_residual_streaming(ridx)
                total += r.weight * ev.value * ev.value
            self._value_cache = total
        return self._value_cache

    def gradient_map(self):
        """请求变量上的一阶梯度（原变量编号 -> 分量）。"""
        if self._grad_cache is None:
            grad = {v: 0.0 for v in self._vars}
            for ridx in self._evidence.relevant:
                r = self._graph._residuals[ridx]
                ev = self._eval_residual(ridx)
                if ev.value == 0.0:
                    continue
                w2v = 2.0 * r.weight * ev.value
                for v in r.support:
                    if v in self._varset:
                        g = ev.grad.get(v)
                        if g:
                            grad[v] += w2v * g
            self._grad_cache = grad
        return dict(self._grad_cache)

    def gradient(self):
        """请求变量上的一阶梯度（按请求序）。"""
        grad = self.gradient_map()
        return tuple(grad[v] for v in self._vars)

    # -- 子块读回 ---------------------------------------------------------

    def _gate_causes_entry(self, r, ev, i, j):
        causes = []
        for gate in ev.gates:
            rows = set(gate.partials) if gate.top_level else r.support
            if i in rows and j in rows:
                if gate.top_level:
                    pi = gate.partials[i]
                    pj = gate.partials[j]
                    toward = 2.0 * r.weight * max(0.0, pj) * pi
                    away = 2.0 * r.weight * min(0.0, pj) * pi
                else:
                    toward = away = None
                causes.append(
                    UndeterminedCause(
                        r.rid, "positive_part_threshold", tuple(sorted(rows)), toward, away
                    )
                )
        return causes

    def _compute_entry(self, i, j, key):
        structural = key in self._evidence.pairs
        total = 0.0
        causes = []
        for ridx in self._evidence.adjacency.get(i, ()):
            r = self._graph._residuals[ridx]
            if j not in r.support:
                continue
            ev = self._eval_residual(ridx)
            flagged = self._gate_causes_entry(r, ev, i, j)
            if flagged:
                causes.extend(flagged)
                continue
            gi = ev.grad.get(i, 0.0)
            gj = ev.grad.get(j, 0.0)
            hij = ev.hess.get(key, 0.0)
            total += 2.0 * r.weight * (gi * gj + ev.value * hij)
        if causes:
            return ("undetermined", None, structural, total, tuple(causes))
        return ("determined", total, structural, total, ())

    def _entry(self, i, j, rp, cp) -> EntryResult:
        key = (i, j)
        result = self._entry_cache.get(key)
        if result is None:
            canon = _canon(i, j)
            payload = self._payload_cache.get(canon)
            if payload is None:
                self._spend(2 * _ENTRY_BYTES, "子块条目")
                payload = self._compute_entry(i, j, canon)
                self._payload_cache[canon] = payload
            status, value, structural, part, causes = payload
            result = EntryResult(i, j, rp, cp, status, value, structural, part, causes)
            self._entry_cache[key] = result
        return result

    def hessian_entry(self, a, b) -> EntryResult:
        """按原变量编号读回单个二阶条目。"""
        i, j = _as_var(a), _as_var(b)
        if i not in self._varset or j not in self._varset:
            raise ContractError("子块坐标超出请求变量范围")
        return self._entry(i, j, self._pos.get(i), self._pos.get(j))

    def subblock(self, coordinates=None):
        """以稀疏坐标读回指定子块。

        coordinates 为 [行, 列] 对：整数对按 requests.vars 的位置解释，
        或用 [参数名, 下标] 对直接指定原变量。缺省用请求中的 coordinates。
        """
        coords = coordinates if coordinates is not None else self._req_coordinates
        if coords is None:
            raise ContractError("未给出子块坐标 coordinates")
        normalized = []
        for c in coords:
            if isinstance(c[0], int):
                rp, cp = _check_coordinate(c, len(self._vars))
                i, j = self._vars[rp], self._vars[cp]
            else:
                i, j = _as_var(c[0]), _as_var(c[1])
                if i not in self._varset or j not in self._varset:
                    raise ContractError("子块坐标超出请求变量范围")
                rp, cp = self._pos.get(i), self._pos.get(j)
            normalized.append((i, j, rp, cp))
        key = tuple((i, j) for i, j, _, _ in normalized)
        block = self._block_cache.get(key)
        if block is None:
            block = tuple(self._entry(i, j, rp, cp) for i, j, rp, cp in normalized)
            self._block_cache[key] = block
        return block

    # -- 方向积 -----------------------------------------------------------

    def _product(self, direction) -> DirectionResult:
        hit = self._product_cache.get(direction)
        if hit is not None:
            return hit
        self._spend(_PRODUCT_BASE_BYTES + len(self._vars) * _ENTRY_BYTES, "方向积")
        dmap = {v: direction[k] for k, v in enumerate(self._vars) if direction[k] != 0.0}
        acc = {v: 0.0 for v in self._vars}
        flags = {v: [] for v in self._vars}
        for ridx in self._evidence.relevant:
            r = self._graph._residuals[ridx]
            ev = self._eval_residual(ridx)
            w2 = 2.0 * r.weight
            for gate in ev.gates:
                dg = sum(gate.partials.get(v, 0.0) * dv for v, dv in dmap.items())
                if dg == 0.0:
                    continue  # 零方向或保持门值不动的方向：两侧一致
                rows = set(gate.partials) if gate.top_level else r.support
                for v in rows:
                    if v not in self._varset:
                        continue
                    if gate.top_level:
                        pi = gate.partials[v]
                        toward = w2 * max(0.0, dg) * pi
                        away = w2 * min(0.0, dg) * pi
                    else:
                        toward = away = None
                    flags[v].append(
                        UndeterminedCause(
                            r.rid,
                            "positive_part_threshold",
                            tuple(sorted(rows)),
                            toward,
                            away,
                        )
                    )
            gd = sum(ev.grad.get(v, 0.0) * dv for v, dv in dmap.items())
            hd = None
            for (u, vv), m in ev.hess.items():
                du = dmap.get(u)
                if du is not None:
                    hd = hd or {}
                    hd[vv] = hd.get(vv, 0.0) + m * du
                dvv = dmap.get(vv)
                if dvv is not None and vv != u:
                    hd = hd or {}
                    hd[u] = hd.get(u, 0.0) + m * dvv
            for v in r.support:
                if v not in self._varset:
                    continue
                contrib = ev.grad.get(v, 0.0) * gd
                if hd is not None and ev.value != 0.0:
                    contrib += ev.value * hd.get(v, 0.0)
                if contrib:
                    acc[v] += w2 * contrib
        components = []
        for k, v in enumerate(self._vars):
            if flags[v]:
                components.append(
                    ComponentResult(v, k, "undetermined", None, acc[v], tuple(flags[v]))
                )
            else:
                components.append(
                    ComponentResult(v, k, "determined", acc[v], acc[v], ())
                )
        needed_ok = all(not flags[v] for v in dmap)
        curvature = None
        if needed_ok:
            curvature = sum(dv * acc[v] for v, dv in dmap.items())
        status = "determined" if not any(flags.values()) else "partial"
        result = DirectionResult(direction, status, tuple(components), curvature)
        self._product_cache[direction] = result
        return result

    def _as_direction_seq(self, d):
        if isinstance(d, Mapping):
            seq = [0.0] * len(self._vars)
            for key, coef in d.items():
                v = _as_var(key)
                if v not in self._pos:
                    raise ContractError(f"方向含请求范围外的变量：{v}")
                seq[self._pos[v]] = float(coef)
            return tuple(seq)
        return _check_direction(d, len(self._vars))

    def direction_products(self, directions=None):
        """批量乘若干方向：每个方向给出 H·d 的逐分量结果。

        方向按 requests.vars 的分量排列，或以 {变量: 系数} 映射给出。
        缺省用请求中的 directions。
        """
        dirs = directions if directions is not None else self._req_directions
        if dirs is None:
            raise ContractError("未给出方向 directions")
        normalized = tuple(self._as_direction_seq(d) for d in dirs)
        batch = self._batch_cache.get(normalized)
        if batch is None:
            batch = tuple(self._product(d) for d in normalized)
            self._batch_cache[normalized] = batch
        return batch

    # -- 诊断 -------------------------------------------------------------

    def nonsmooth_scopes(self):
        """未定义二阶的分量及其受影响变量作用域（原变量编号）。"""
        if self._scopes_cache is None:
            out = []
            for ridx, r in enumerate(self._graph._residuals):
                if not r.gated or r.rid not in self._enabled:
                    continue
                ev = self._eval_residual_streaming(ridx)
                for gate in ev.gates:
                    scope = (
                        tuple(sorted(gate.partials))
                        if gate.top_level
                        else tuple(sorted(r.support))
                    )
                    out.append(
                        ScopeReport(
                            r.rid,
                            "positive_part_threshold",
                            scope,
                            gate.gate_value,
                            gate.top_level,
                            any(v in self._varset for v in scope),
                        )
                    )
            self._scopes_cache = tuple(out)
        return self._scopes_cache

    def run_request(self):
        """按请求一次性交付：子块读回 + 批量方向积。"""
        out = {}
        if self._req_coordinates is not None:
            out["block"] = self.subblock()
        if self._req_directions is not None:
            out["products"] = self.direction_products()
        return out

    def cached_results(self):
        """按求值实例查回已算出的旧结果（不触发新计算）。"""
        return {
            "token": self.token,
            "value": self._value_cache,
            "gradient": None
            if self._grad_cache is None
            else tuple(self._grad_cache[v] for v in self._vars),
            "entries": dict(self._entry_cache),
            "products": dict(self._product_cache),
            "stats": self.stats(),
        }


def _check_direction(d, n):
    seq = tuple(float(x) for x in d)
    if len(seq) != n:
        raise ContractError(f"方向维数 {len(seq)} 与请求变量数 {n} 不一致")
    return seq


def _check_coordinate(c, n):
    row, col = int(c[0]), int(c[1])
    if not (0 <= row < n and 0 <= col < n):
        raise ContractError(f"子块坐标 {[row, col]} 超出请求变量范围 {n}")
    return (row, col)


# ---------------------------------------------------------------------------
# 能量图
# ---------------------------------------------------------------------------


class EnergyGraph:
    """由交换契约构建的可微能量图。"""

    def __init__(self, record, *, label="g0", budget=None, boundary_tol=DEFAULT_BOUNDARY_TOL):
        self.label = label
        self._tol = float(boundary_tol)
        params = record.get("parameters")
        if not isinstance(params, Mapping) or not params:
            raise ContractError("parameters 应为非空映射")
        self._parameters = {}
        for name, values in params.items():
            vals = [float(x) for x in values]
            if not vals:
                raise ContractError(f"参数 {name!r} 为空")
            self._parameters[name] = vals
        self._views = {}
        for name, item in (record.get("views") or {}).items():
            param, indices = item
            if param not in self._parameters:
                raise ContractError(f"视图 {name!r} 引用未知参数 {param!r}")
            size = len(self._parameters[param])
            view = []
            for raw in indices:
                idx = int(raw)
                if not 0 <= idx < size:
                    raise ContractError(f"视图 {name!r} 下标 {idx} 越界")
                view.append(Var(param, idx))
            if not view:
                raise ContractError(f"视图 {name!r} 为空")
            self._views[name] = tuple(view)
        self._residuals = []
        seen = set()
        for item in record.get("residuals") or []:
            rid = item.get("id")
            if not isinstance(rid, str) or not rid:
                raise ContractError("残差缺少标识 id")
            if rid in seen:
                raise ContractError(f"残差标识重复：{rid!r}")
            seen.add(rid)
            loss = item.get("loss", "square")
            if loss != "square":
                raise ContractError(f"未知损失 {loss!r}：当前支持 square")
            expr = parse_expr(item.get("expr"), self._views)
            self._residuals.append(
                Residual(
                    rid,
                    expr,
                    float(item.get("weight", 1.0)),
                    loss,
                    _support(expr),
                    _has_gate(expr),
                )
            )
        if not self._residuals:
            raise ContractError("residuals 为空")
        # 静态倒排：原变量 -> 引用它的残差（建图一次，各实例复用）
        by_var = {}
        for ridx, r in enumerate(self._residuals):
            for v in r.support:
                by_var.setdefault(v, []).append(ridx)
        self._by_var = {v: tuple(lst) for v, lst in by_var.items()}
        self._states = {}
        for st in record.get("states") or []:
            sid = st.get("id")
            if not isinstance(sid, str) or not sid:
                raise ContractError("状态缺少标识 id")
            self._states[sid] = st
        scale = record.get("scale") or record.get("limits") or {}
        self.scale_info = dict(scale)
        self._budget = budget if budget is not None else Budget(scale.get("scratch_bytes"))
        self._requests = record.get("requests")
        self._instances = {}
        self._structures = {}

    # -- 基本信息 ---------------------------------------------------------

    @property
    def residuals(self):
        return tuple(self._residuals)

    @property
    def default_request(self):
        return self._requests

    @property
    def all_vars(self):
        """全部原变量（按参数名、物理变量号排序）。"""
        return tuple(
            sorted(Var(p, i) for p, vals in self._parameters.items() for i in range(len(vals)))
        )

    @property
    def instances(self):
        """已创建的求值实例：token -> Evaluation。"""
        return {inst.token: inst for inst in self._instances.values()}

    def instance(self, token) -> Evaluation:
        """按 token 查回求值实例及其缓存的旧结果。"""
        for inst in self._instances.values():
            if inst.token == token:
                return inst
        raise KeyError(f"未知求值实例：{token!r}")

    def _check_var(self, v: Var):
        vals = self._parameters.get(v.param)
        if vals is None:
            raise ContractError(f"未知参数：{v.param!r}")
        if not 0 <= v.index < len(vals):
            raise ContractError(f"变量越界：{v}")

    # -- 求值实例 ---------------------------------------------------------

    def evaluation(self, state=None, *, point=None, enabled=None, variables=None, request=None):
        """绑定求值点、启用掩码与请求变量范围，返回求值实例。

        state 为契约中的状态 id 或状态映射；point 可用 {(参数名, 下标): 值}
        进一步覆盖；enabled 缺省取状态启用表、再缺省为全部残差；variables
        缺省取请求的 vars。相同内容的调用返回同一实例。
        """
        params = {k: list(v) for k, v in self._parameters.items()}
        en = None
        if state is not None:
            if isinstance(state, str):
                st = self._states.get(state)
                if st is None:
                    raise ContractError(f"未知状态：{state!r}")
            else:
                st = dict(state)
            for key, val in st.items():
                if key in ("id", "enabled"):
                    continue
                m = _OVERRIDE_RE.match(key)
                if not m:
                    raise ContractError(f"无法识别的状态覆盖项：{key!r}")
                param, idx = m.group(1), int(m.group(2))
                if param not in params:
                    raise ContractError(f"状态覆盖引用未知参数：{param!r}")
                if not 0 <= idx < len(params[param]):
                    raise ContractError(f"状态覆盖下标越界：{key!r}")
                params[param][idx] = float(val)
            if en is None and "enabled" in st:
                en = list(st["enabled"])
        if point:
            for key, val in point.items():
                v = _as_var(key)
                self._check_var(v)
                params[v.param][v.index] = float(val)
        if enabled is not None:
            en = list(enabled)
        if en is None:
            en = [r.rid for r in self._residuals]
        known = {r.rid for r in self._residuals}
        bad = [x for x in en if x not in known]
        if bad:
            raise ContractError(f"启用表含未知残差：{bad}")
        # 方向与坐标按 requests.vars 排列：只有变量范围同样来自该请求时才继承
        if request is not None:
            req = request
            inherit = True
        elif variables is None:
            req = self._requests or {}
            inherit = True
        else:
            req = {}
            inherit = False
        if variables is None:
            variables = req.get("vars")
        if not variables:
            raise ContractError("缺少请求变量范围（vars）")
        vars_ = tuple(_as_var(v) for v in variables)
        if len(set(vars_)) != len(vars_):
            raise ContractError("请求变量范围含重复变量")
        for v in vars_:
            self._check_var(v)
        directions = req.get("directions") if inherit else None
        if directions is not None:
            directions = tuple(_check_direction(d, len(vars_)) for d in directions)
        coordinates = req.get("coordinates") if inherit else None
        if coordinates is not None:
            coordinates = tuple(_check_coordinate(c, len(vars_)) for c in coordinates)
        key = (
            tuple((p, tuple(vals)) for p, vals in sorted(params.items())),
            tuple(sorted(en)),
            vars_,
        )
        inst = self._instances.get(key)
        if inst is None:
            token = f"{self.label}#{len(self._instances)}"
            inst = Evaluation(self, params, en, vars_, directions, coordinates, token)
            self._instances[key] = inst
        return inst


class Session:
    """工作会话：管理多张能量图，支持按实例查回旧结果。"""

    def __init__(self, *, boundary_tol=DEFAULT_BOUNDARY_TOL):
        self._tol = float(boundary_tol)
        self.graphs = {}

    def build_graph(self, record, *, label=None, budget=None) -> EnergyGraph:
        """按交换契约构建能量图；换一张图即再调用一次。"""
        label = label or f"g{len(self.graphs)}"
        if label in self.graphs:
            raise ContractError(f"图标签重复：{label!r}")
        graph = EnergyGraph(record, label=label, budget=budget, boundary_tol=self._tol)
        self.graphs[label] = graph
        return graph

    def instance(self, token) -> Evaluation:
        """跨图按 token 查回求值实例。"""
        label, _, _ = token.partition("#")
        graph = self.graphs.get(label)
        if graph is None:
            raise KeyError(f"未知求值实例：{token!r}")
        return graph.instance(token)


def main():
    """打印本地交换样本各状态的子块与方向积。"""
    from sparse_second_order_input import load_record

    record = load_record()
    session = Session()
    graph = session.build_graph(record, label="demo")
    out = []
    for state in record["states"]:
        inst = graph.evaluation(state=state, request=record["requests"])
        delivered = inst.run_request()
        out.append(
            {
                "状态": state["id"],
                "实例": inst.token,
                "能量": inst.value(),
                "梯度": {f"{v.param}[{v.index}]": g for v, g in inst.gradient_map().items()},
                "不可微作用域": [s.to_dict() for s in inst.nonsmooth_scopes()],
                "子块": [e.to_dict() for e in delivered["block"]],
                "方向积": [d.to_dict() for d in delivered["products"]],
                "统计": inst.stats(),
            }
        )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
