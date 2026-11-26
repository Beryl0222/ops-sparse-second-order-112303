"""稀疏二阶算子的行为测试。

数值期望均由契约样本手工推导：
能量 = 启用项 weight × residual² 之和；H[i,j] 按列坐标方向上的梯度分量变化。
"""
import unittest

from sparse_second_order_input import load_record
from sparse_second_order import (
    Budget,
    BudgetExceededError,
    ContractError,
    Session,
    Var,
)

X0, X2, X3, Z0, Z1 = Var("x", 0), Var("x", 2), Var("x", 3), Var("z", 0), Var("z", 1)


def make_session(**graph_kw):
    session = Session()
    return session, session.build_graph(load_record(), label="t", **graph_kw)


class SparseTestCase(unittest.TestCase):
    def assertValuesAlmost(self, got, want, places=7):
        """逐分量比较浮点结果，允许求和顺序带来的尾差。"""
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            if w is None:
                self.assertIsNone(g)
            else:
                self.assertAlmostEqual(g, w, places=places)


class GraphBuildTests(SparseTestCase):
    def test_alias_support(self):
        """同一原变量经多视图进入多个残差，静态支撑按原变量编号。"""
        _, graph = make_session()
        by_id = {r.rid: r for r in graph.residuals}
        self.assertEqual(by_id["pair"].support, frozenset({X0, X2}))
        # mirror[0] 即 x[2]：cancel 与 pair 是同一相互作用
        self.assertEqual(by_id["cancel"].support, frozenset({X0, X2}))
        self.assertEqual(by_id["gate"].support, frozenset({X2, Z0}))
        self.assertEqual(by_id["side"].support, frozenset({X3, Z1}))
        self.assertTrue(by_id["gate"].gated)
        self.assertFalse(by_id["pair"].gated)

    def test_bad_contract(self):
        _, graph = make_session()
        record = load_record()
        bad_loss = dict(record, residuals=[dict(record["residuals"][0], loss="huber")])
        with self.assertRaises(ContractError):
            Session().build_graph(bad_loss)
        bad_op = dict(record, residuals=[dict(record["residuals"][0], expr=["mystery", ["p", 0]])])
        with self.assertRaises(ContractError):
            Session().build_graph(bad_op)
        bad_view = dict(record, residuals=[dict(record["residuals"][0], expr=["nope", 0])])
        with self.assertRaises(ContractError):
            Session().build_graph(bad_view)
        bad_index = dict(record, views={"p": ["x", [0, 9]]})
        with self.assertRaises(ContractError):
            Session().build_graph(bad_index)

    def test_bad_request(self):
        _, graph = make_session()
        record = load_record()
        with self.assertRaises(ContractError):
            graph.evaluation(state="left", variables=[["x", 0], ["x", 0]])
        with self.assertRaises(ContractError):
            graph.evaluation(state="left", variables=[["x", 9]])
        with self.assertRaises(ContractError):
            graph.evaluation(state="left", variables=[["x", 0]], enabled=["pair", "ghost"])
        with self.assertRaises(ContractError):
            graph.evaluation(state="ghost", request=record["requests"])
        inst = graph.evaluation(state="left", request=record["requests"])
        with self.assertRaises(ContractError):
            inst.direction_products([[1.0, 0.0]])
        with self.assertRaises(ContractError):
            inst.subblock([[0, 4]])
        with self.assertRaises(ContractError):
            inst.hessian_entry(["x", 1], ["x", 0])  # x[1] 不在请求范围


class LeftStateTests(SparseTestCase):
    """left：门关闭（g=-0.1），pair 与 cancel 数值抵消但结构保留。"""

    @classmethod
    def setUpClass(cls):
        _, cls.graph = make_session()
        cls.inst = cls.graph.evaluation(state="left", request=load_record()["requests"])

    def test_value_and_gradient(self):
        self.assertAlmostEqual(self.inst.value(), 0.144)
        self.assertEqual(len(self.inst.gradient()), 4)
        for got, want in zip(self.inst.gradient(), (0.0, 0.0, 0.48, 0.0)):
            self.assertAlmostEqual(got, want)

    def test_block(self):
        entries = {(e.row, e.col): e for e in self.inst.subblock()}
        e01 = entries[(X0, X2)]
        # 数值抵消为真零，但结构证据保留交互可能出现
        self.assertEqual(e01.status, "determined")
        self.assertAlmostEqual(e01.value, 0.0)
        self.assertTrue(e01.structural)
        e13 = entries[(X2, Z0)]
        # 门关闭：结构上不存在该交互，是真零
        self.assertEqual(e13.status, "determined")
        self.assertAlmostEqual(e13.value, 0.0)
        self.assertFalse(e13.structural)
        e22 = entries[(X3, X3)]
        self.assertEqual(e22.status, "determined")
        self.assertAlmostEqual(e22.value, 0.8)

    def test_products(self):
        d0, d1, d2, d3 = self.inst.direction_products()
        self.assertEqual(d0.status, "determined")
        self.assertValuesAlmost(d0.values(), (0.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(d0.curvature, 0.0)
        self.assertEqual(d1.status, "determined")
        self.assertValuesAlmost(d1.values(), (0.0, 0.0, 0.8, 0.0))
        self.assertAlmostEqual(d1.curvature, 0.8)
        self.assertValuesAlmost(d2.values(), (0.0, 0.0, 0.0, 0.0))
        self.assertValuesAlmost(d3.values(), (0.0, 0.0, 0.0, 0.0))  # 零方向确定可用
        self.assertTrue(all(d.usable for d in (d0, d1, d2, d3)))

    def test_no_nonsmooth_scope(self):
        self.assertEqual(self.inst.nonsmooth_scopes(), ())

    def test_structure_keeps_cancelled_pair(self):
        pairs = set(self.inst.structure_pairs())
        self.assertEqual(
            pairs,
            {
                (X0, X0),
                (X0, X2),
                (X2, X2),
                (X3, X3),
            },
        )


class EdgeStateTests(SparseTestCase):
    """edge：门恰好位于门槛，作用域 {x2, z0} 内二阶不可定。"""

    @classmethod
    def setUpClass(cls):
        _, cls.graph = make_session()
        cls.inst = cls.graph.evaluation(state="edge", request=load_record()["requests"])

    def test_value_and_gradient(self):
        self.assertAlmostEqual(self.inst.value(), 0.144)
        for got, want in zip(self.inst.gradient(), (0.0, 0.0, 0.48, 0.0)):
            self.assertAlmostEqual(got, want)

    def test_scope_report(self):
        (scope,) = self.inst.nonsmooth_scopes()
        self.assertEqual(scope.residual, "gate")
        self.assertEqual(scope.kind, "positive_part_threshold")
        self.assertEqual(scope.scope, (X2, Z0))
        self.assertTrue(scope.top_level)
        self.assertTrue(scope.reaches_request)
        self.assertAlmostEqual(scope.gate_value, 0.0, delta=1e-12)

    def test_block(self):
        entries = {(e.row, e.col): e for e in self.inst.subblock()}
        e01 = entries[(X0, X2)]
        # 远离门作用域的已确定项照常交付（数值抵消的真零）
        self.assertEqual(e01.status, "determined")
        self.assertAlmostEqual(e01.value, 0.0)
        self.assertTrue(e01.structural)
        e13 = entries[(X2, Z0)]
        # 行列均在门作用域：不可定，不能默认采用某侧二阶导数
        self.assertEqual(e13.status, "undetermined")
        self.assertIsNone(e13.value)
        self.assertTrue(e13.structural)
        self.assertAlmostEqual(e13.determined_part, 0.0)
        (cause,) = e13.causes
        self.assertEqual(cause.residual, "gate")
        self.assertEqual(cause.scope, (X2, Z0))
        self.assertAlmostEqual(cause.toward, 0.0)   # 沿 +e_z0 一侧
        self.assertAlmostEqual(cause.away, -2.0)    # 沿 -e_z0 一侧
        e22 = entries[(X3, X3)]
        self.assertEqual(e22.status, "determined")
        self.assertAlmostEqual(e22.value, 0.8)

    def test_products(self):
        d0, d1, d2, d3 = self.inst.direction_products()
        # d0 = e_x2：门值随之移动，作用域内分量不可定，其余照常
        self.assertEqual(d0.status, "partial")
        self.assertFalse(d0.usable)
        self.assertIsNone(d0.curvature)
        comp = {c.var: c for c in d0.components}
        self.assertEqual(comp[X0].status, "determined")
        self.assertAlmostEqual(comp[X0].value, 0.0)
        self.assertEqual(comp[X3].status, "determined")
        self.assertAlmostEqual(comp[X3].value, 0.0)
        self.assertEqual(comp[X2].status, "undetermined")
        self.assertAlmostEqual(comp[X2].causes[0].toward, 2.0)
        self.assertAlmostEqual(comp[X2].causes[0].away, 0.0)
        self.assertEqual(comp[Z0].status, "undetermined")
        self.assertAlmostEqual(comp[Z0].causes[0].toward, -2.0)
        self.assertAlmostEqual(comp[Z0].causes[0].away, 0.0)
        # d1 = e_x3：与门正交，完全确定
        self.assertEqual(d1.status, "determined")
        self.assertValuesAlmost(d1.values(), (0.0, 0.0, 0.8, 0.0))
        self.assertAlmostEqual(d1.curvature, 0.8)
        # d2 = e_x2 + e_z0：从两边接近都留在门槛上，按同一口径确定
        self.assertEqual(d2.status, "determined")
        self.assertValuesAlmost(d2.values(), (0.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(d2.curvature, 0.0)
        # d3 零方向：同一口径确定
        self.assertEqual(d3.status, "determined")
        self.assertValuesAlmost(d3.values(), (0.0, 0.0, 0.0, 0.0))

    def test_structure_includes_threshold_pair(self):
        pairs = set(self.inst.structure_pairs())
        self.assertIn((X2, Z0), pairs)
        self.assertIn((Z0, Z0), pairs)
        self.assertIn((X0, X2), pairs)


class RightStateTests(SparseTestCase):
    """right：门打开（g=0.3），cancel 被禁用，相互作用净出现。"""

    @classmethod
    def setUpClass(cls):
        _, cls.graph = make_session()
        cls.inst = cls.graph.evaluation(state="right", request=load_record()["requests"])

    def test_value_and_gradient(self):
        self.assertAlmostEqual(self.inst.value(), 0.6948)
        for got, want in zip(self.inst.gradient(), (1.152, 2.136, 0.48, -0.6)):
            self.assertAlmostEqual(got, want)

    def test_block(self):
        entries = {(e.row, e.col): e for e in self.inst.subblock()}
        self.assertAlmostEqual(entries[(X0, X2)].value, 3.84)
        self.assertAlmostEqual(entries[(X2, Z0)].value, -2.0)
        self.assertAlmostEqual(entries[(X3, X3)].value, 0.8)
        self.assertTrue(all(e.status == "determined" for e in entries.values()))

    def test_products(self):
        d0, d1, d2, d3 = self.inst.direction_products()
        self.assertValuesAlmost(d0.values(), (3.84, 4.56, 0.0, -2.0))
        self.assertAlmostEqual(d0.curvature, 4.56)
        self.assertValuesAlmost(d1.values(), (0.0, 0.0, 0.8, 0.0))
        self.assertValuesAlmost(d2.values(), (3.84, 2.56, 0.0, 0.0))
        self.assertAlmostEqual(d2.curvature, 2.56)
        self.assertValuesAlmost(d3.values(), (0.0, 0.0, 0.0, 0.0))
        self.assertTrue(all(d.usable for d in (d0, d1, d2, d3)))

    def test_no_nonsmooth_scope(self):
        self.assertEqual(self.inst.nonsmooth_scopes(), ())


class StructureReuseTests(SparseTestCase):
    def test_same_signature_reuses_evidence(self):
        _, graph = make_session()
        req = load_record()["requests"]
        left = graph.evaluation(state="left", request=req)
        # 同一图上的连续求值点：掩码与软阈分支未变，结构证据直接复用
        moved = graph.evaluation(state="left", point={("x", 0): 0.9}, request=req)
        self.assertIsNot(left, moved)
        self.assertIs(left.structure, moved.structure)
        self.assertTrue(moved.stats()["structure_reused"])
        # 数值按各自实例计算：pair 与 cancel 依然抵消
        self.assertAlmostEqual(moved.hessian_entry(X0, X0).value, 0.0)

    def test_mask_and_threshold_change_rediscover(self):
        _, graph = make_session()
        req = load_record()["requests"]
        left = graph.evaluation(state="left", request=req)
        edge = graph.evaluation(state="edge", request=req)
        right = graph.evaluation(state="right", request=req)
        # 阈值跨越（left→edge）与掩码变化（edge→right）都得到新结构
        self.assertIsNot(left.structure, edge.structure)
        self.assertIsNot(edge.structure, right.structure)
        self.assertNotEqual(left.structure_signature, edge.structure_signature)
        self.assertNotEqual(edge.structure_signature, right.structure_signature)
        self.assertNotIn((X2, Z0), set(left.structure_pairs()))
        self.assertIn((X2, Z0), set(edge.structure_pairs()))

    def test_values_never_merge_across_instances(self):
        """新出现的交叉项不沿旧实例的试探身份相加。"""
        _, graph = make_session()
        req = load_record()["requests"]
        left = graph.evaluation(state="left", request=req)
        edge = graph.evaluation(state="edge", request=req)
        right = graph.evaluation(state="right", request=req)
        left_d0, *_ = left.direction_products()
        edge_d0, *_ = edge.direction_products()
        right_d0, *_ = right.direction_products()
        # 旧实例缓存的结果不被后续求值改写
        self.assertValuesAlmost(left_d0.values(), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(edge_d0.status, "partial")
        self.assertValuesAlmost(right_d0.values(), (3.84, 4.56, 0.0, -2.0))
        # 交叉项 (x2, z0)：left 真零、edge 不可定、right 为 -2，各自独立
        self.assertAlmostEqual(left.hessian_entry(X2, Z0).value, 0.0)
        self.assertIsNone(edge.hessian_entry(X2, Z0).value)
        self.assertAlmostEqual(right.hessian_entry(X2, Z0).value, -2.0)


class InstanceRegistryTests(SparseTestCase):
    def test_content_keyed_instance(self):
        session, graph = make_session()
        req = load_record()["requests"]
        a = graph.evaluation(state="edge", request=req)
        b = graph.evaluation(state="edge", request=req)
        self.assertIs(a, b)
        # 按当时的求值实例查回旧结果
        again = session.instance(a.token)
        self.assertIs(again, a)
        block_first = a.subblock()
        self.assertIs(a.subblock(), block_first)
        self.assertIs(a.hessian_entry(X2, Z0), a.hessian_entry(X2, Z0))
        products_first = a.direction_products()
        self.assertIs(a.direction_products(), products_first)
        cached = a.cached_results()
        self.assertEqual(cached["token"], a.token)
        self.assertTrue(cached["entries"])
        with self.assertRaises(KeyError):
            session.instance("t#999")

    def test_switch_graph_point_and_range(self):
        session, graph = make_session()
        record = load_record()
        # 换一张图：权重改变，互不干扰
        other_record = dict(record)
        other_record["residuals"] = [
            dict(r, weight=3) if r["id"] == "side" else r for r in record["residuals"]
        ]
        other = session.build_graph(other_record, label="other")
        req = record["requests"]
        base_inst = graph.evaluation(state="left", request=req)
        other_inst = other.evaluation(state="left", request=req)
        self.assertAlmostEqual(base_inst.hessian_entry(X3, X3).value, 0.8)
        self.assertAlmostEqual(other_inst.hessian_entry(X3, X3).value, 6.0)
        self.assertIsNot(base_inst, other_inst)
        # 改参数范围：只取门作用域两个变量
        narrow = graph.evaluation(state="edge", variables=[["x", 2], ["z", 0]])
        entry = narrow.hessian_entry(X2, Z0)
        self.assertEqual(entry.status, "undetermined")
        (scope,) = narrow.nonsmooth_scopes()
        self.assertTrue(scope.reaches_request)

    def test_residual_evals_shared_between_outputs(self):
        """同一实例上子块与方向积复用残差求值，不重复计算。"""
        _, graph = make_session()
        inst = graph.evaluation(state="right", request=load_record()["requests"])
        inst.subblock()
        after_block = inst.stats()["residual_evals"]
        inst.direction_products()
        self.assertEqual(inst.stats()["residual_evals"], after_block)
        inst.gradient()
        self.assertEqual(inst.stats()["residual_evals"], after_block)


class BudgetTests(SparseTestCase):
    def test_budget_exceeded(self):
        session = Session()
        graph = session.build_graph(
            load_record(), label="b", budget=Budget(scratch_bytes=10)
        )
        with self.assertRaises(BudgetExceededError):
            graph.evaluation(state="left", request=load_record()["requests"])

    def test_default_budget_from_scale(self):
        _, graph = make_session()
        inst = graph.evaluation(state="right", request=load_record()["requests"])
        inst.run_request()
        self.assertLess(inst.stats()["scratch_bytes"], 33554432)


REACH_RECORD = {
    "parameters": {"x": [1.0, 2.0, 3.0, 4.0]},
    "views": {"u": ["x", [0, 1]], "w": ["x", [2, 3]]},
    "residuals": [
        {"id": "near", "expr": ["product", ["u", 0], ["u", 1]], "loss": "square", "weight": 1},
        {"id": "far", "expr": ["product", ["w", 0], ["w", 1]], "loss": "square", "weight": 1},
    ],
    "states": [{"id": "s", "enabled": ["near", "far"]}],
    "requests": {"vars": [["x", 0]], "directions": [[1]], "coordinates": [[0, 0]]},
}


class ReachabilityTests(SparseTestCase):
    def test_only_reachable_residuals_are_evaluated(self):
        graph = Session().build_graph(REACH_RECORD)
        inst = graph.evaluation(state="s", request=REACH_RECORD["requests"])
        (entry,) = inst.subblock()
        self.assertAlmostEqual(entry.value, 8.0)  # 2 * (x1)^2 = 2*4
        self.assertEqual(inst.stats()["candidates"], 1)
        self.assertEqual(inst.stats()["residual_evals"], 1)  # far 不可达，未求值
        # 能量需要全部启用分量：far 流式求值但不进缓存
        self.assertAlmostEqual(inst.value(), 1 * 4 + 1 * 144)
        self.assertEqual(inst.stats()["residual_evals"], 2)
        self.assertEqual(inst.stats()["cached_evals"], 1)


SELF_PRODUCT_RECORD = {
    "parameters": {"x": [3.0]},
    "views": {"u": ["x", [0]], "w": ["x", [0]]},
    "residuals": [
        {"id": "sp", "expr": ["product", ["u", 0], ["w", 0]], "loss": "square", "weight": 0.5}
    ],
    "states": [{"id": "s", "enabled": ["sp"]}],
    "requests": {"vars": [["x", 0]], "directions": [[2]], "coordinates": [[0, 0]]},
}

ZERO_WEIGHT_RECORD = {
    "parameters": {"x": [1.0, 2.0]},
    "views": {"u": ["x", [0, 1]]},
    "residuals": [
        {"id": "zw", "expr": ["product", ["u", 0], ["u", 1]], "loss": "square", "weight": 0}
    ],
    "states": [{"id": "s", "enabled": ["zw"]}],
    "requests": {"vars": [["x", 0], ["x", 1]], "coordinates": [[0, 1]]},
}


class SemanticsTests(SparseTestCase):
    def test_aliased_self_product(self):
        """同一原变量经两个视图相乘：ρ = x²，二阶按别名正确合并。"""
        graph = Session().build_graph(SELF_PRODUCT_RECORD)
        inst = graph.evaluation(state="s", request=SELF_PRODUCT_RECORD["requests"])
        self.assertAlmostEqual(inst.value(), 0.5 * 81)
        self.assertAlmostEqual(inst.gradient()[0], 54.0)
        (entry,) = inst.subblock()
        self.assertAlmostEqual(entry.value, 54.0)  # 12 * w * x² = 12*0.5*9
        (prod,) = inst.direction_products()
        self.assertAlmostEqual(prod.values()[0], 108.0)

    def test_zero_weight_residual_not_structural(self):
        graph = Session().build_graph(ZERO_WEIGHT_RECORD)
        inst = graph.evaluation(state="s", request=ZERO_WEIGHT_RECORD["requests"])
        (entry,) = inst.subblock()
        self.assertEqual(entry.status, "determined")
        self.assertAlmostEqual(entry.value, 0.0)
        self.assertFalse(entry.structural)
        self.assertEqual(inst.structure_pairs(), ())

    def test_var_and_mapping_forms(self):
        """坐标与方向也可用原变量形式给出。"""
        _, graph = make_session()
        inst = graph.evaluation(state="right", request=load_record()["requests"])
        (entry,) = inst.subblock([[["x", 0], ["x", 2]]])
        self.assertAlmostEqual(entry.value, 3.84)
        (prod,) = inst.direction_products([{("x", 2): 1.0}])
        self.assertValuesAlmost(prod.values(), (3.84, 4.56, 0.0, -2.0))


if __name__ == "__main__":
    unittest.main()
