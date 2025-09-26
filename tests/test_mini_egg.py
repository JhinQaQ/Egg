import pytest

from .. import (
    EGraph,
    Runner,
    REWRITES,
    COMM_ADD,
    ASSOC_ADD,
    COMM_MUL,
    ASSOC_MUL,
    DISTR_L,
    DISTR_R,
)


class TestMiniEgg:
    def test_build_and_extract_identity(self):
        eg = EGraph()
        expr = (
            "Mul",
            ("Add", ("Num", 1), ("Num", 2)),
            ("Add", ("Num", 3), ("Add", ("Num", 4), ("Num", 5))),
        )
        root = eg.add_term(expr)
        Runner(eg).run(3, rewrites=REWRITES)
        best = eg.extract(root)
        assert best[0] == "Mul"
        assert best[1][0] == "Add"
        assert best[2][0] == "Add"

    def test_commutativity_add(self):
        eg = EGraph()
        a = eg.add_term(("Add", ("Num", 1), ("Num", 2)))
        b = eg.add_term(("Add", ("Num", 2), ("Num", 1)))
        assert eg.uf.find(a) != eg.uf.find(b)
        Runner(eg).run(2, rewrites=[COMM_ADD])
        assert eg.uf.find(a) == eg.uf.find(b)

    def test_commutativity_mul(self):
        eg = EGraph()
        a = eg.add_term(("Mul", ("Num", 3), ("Num", 4)))
        b = eg.add_term(("Mul", ("Num", 4), ("Num", 3)))
        assert eg.uf.find(a) != eg.uf.find(b)
        Runner(eg).run(2, rewrites=[COMM_MUL])
        assert eg.uf.find(a) == eg.uf.find(b)

    def test_associativity_add(self):
        eg = EGraph()
        left = eg.add_term(("Add", ("Add", ("Num", 1), ("Num", 2)), ("Num", 3)))
        right = eg.add_term(("Add", ("Num", 1), ("Add", ("Num", 2), ("Num", 3))))
        assert eg.uf.find(left) != eg.uf.find(right)
        Runner(eg).run(3, rewrites=[ASSOC_ADD])
        assert eg.uf.find(left) == eg.uf.find(right)

    def test_associativity_mul(self):
        eg = EGraph()
        left = eg.add_term(("Mul", ("Mul", ("Num", 2), ("Num", 3)), ("Num", 4)))
        right = eg.add_term(("Mul", ("Num", 2), ("Mul", ("Num", 3), ("Num", 4))))
        assert eg.uf.find(left) != eg.uf.find(right)
        Runner(eg).run(3, rewrites=[ASSOC_MUL])
        assert eg.uf.find(left) == eg.uf.find(right)

    def test_distributivity_left(self):
        eg = EGraph()
        left = eg.add_term(("Mul", ("Num", 5), ("Add", ("Num", 2), ("Num", 3))))
        right = eg.add_term(("Add", ("Mul", ("Num", 5), ("Num", 2)), ("Mul", ("Num", 5), ("Num", 3))))
        assert eg.uf.find(left) != eg.uf.find(right)
        Runner(eg).run(3, rewrites=[DISTR_L])
        assert eg.uf.find(left) == eg.uf.find(right)

    def test_distributivity_right(self):
        eg = EGraph()
        left = eg.add_term(("Mul", ("Add", ("Num", 1), ("Num", 2)), ("Num", 4)))
        right = eg.add_term(("Add", ("Mul", ("Num", 1), ("Num", 4)), ("Mul", ("Num", 2), ("Num", 4))))
        assert eg.uf.find(left) != eg.uf.find(right)
        Runner(eg).run(3, rewrites=[DISTR_R])
        assert eg.uf.find(left) == eg.uf.find(right)

    def test_distributivity_creates_equal_form(self):
        eg = EGraph()
        x = eg.add_term(("Mul", ("Num", 7), ("Add", ("Num", 2), ("Num", 3))))
        y = eg.add_term(("Add", ("Mul", ("Num", 7), ("Num", 2)), ("Mul", ("Num", 7), ("Num", 3))))
        Runner(eg).run(4, rewrites=[DISTR_L, DISTR_R])
        assert eg.uf.find(x) == eg.uf.find(y)

    def test_assoc_then_comm(self):
        eg = EGraph()
        t1 = eg.add_term(("Add", ("Num", 1), ("Add", ("Num", 2), ("Num", 3))))
        t2 = eg.add_term(("Add", ("Add", ("Num", 3), ("Num", 2)), ("Num", 1)))
        Runner(eg).run(5, rewrites=[ASSOC_ADD, COMM_ADD])
        assert eg.uf.find(t1) == eg.uf.find(t2)

    def test_complex_equality_proof(self):
        eg = EGraph()
        left = eg.add_term(("Mul", ("Add", ("Num", 1), ("Num", 2)), ("Add", ("Num", 3), ("Num", 4))))
        right = eg.add_term((
            "Add",
            ("Add", ("Mul", ("Num", 1), ("Num", 3)), ("Mul", ("Num", 1), ("Num", 4))),
            ("Add", ("Mul", ("Num", 2), ("Num", 3)), ("Mul", ("Num", 2), ("Num", 4))),
        ))
        Runner(eg).run(10, rewrites=REWRITES)
        assert eg.uf.find(left) == eg.uf.find(right)

    def test_pattern_matching(self):
        eg = EGraph()
        eg.add_term(("Add", ("Num", 1), ("Num", 2)))
        eg.add_term(("Add", ("Num", 3), ("Num", 4)))
        eg.add_term(("Mul", ("Num", 5), ("Num", 6)))

        pattern = ("Add", "?x", "?y")
        matches = eg.ematch(pattern)
        assert len(matches) == 2

    def test_single_number(self):
        eg = EGraph()
        num = eg.add_term(("Num", 42))
        extracted = eg.extract(num)
        assert extracted == ("Num", 42)

