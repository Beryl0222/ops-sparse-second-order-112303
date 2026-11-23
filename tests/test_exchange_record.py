"""检查交换数据中用于回查的事实。"""
import unittest

import numpy as np

from sparse_second_order_input import load_record


class ExchangeRecordTests(unittest.TestCase):
    def test_exchange_contract(self):
        record = load_record()
        self.assertTrue(record["views"]["p"][1][2] == record["views"]["tail"][1][0] == record["views"]["mirror"][1][0])
        self.assertTrue(np.isclose(record["states"][1]["x_index2"] - record["parameters"]["z"][0] - 0.2,0,atol=1e-12))
        self.assertTrue(record["residuals"][0]["weight"] == -record["residuals"][-1]["weight"] and "cancel" not in record["states"][-1]["enabled"])
        self.assertTrue(record["requests"]["directions"][-1] == [0,0,0,0] and record["requests"]["directions"][-2] == [0,1,0,1])
