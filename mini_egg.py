# mini_egg.py — tiny e-graph with rebuilding, rewrites, extraction

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import unittest

# -------------------------------------------------------------------
# Term & helpers
# Terms are tuples:
#   ("Num", n)                      # literal
#   ("Add", t1, t2) / ("Mul", t1, t2)
#
# Pattern variables are strings that start with '?', e.g., '?x'

Op = str
Term = Tuple[Any, ...]
EClassId = int
ENodeKey = Tuple[Op, Tuple[EClassId, ...]]  # canonical (op, child leaders)

def is_var(x: Any) -> bool:
    return isinstance(x, str) and x.startswith("?")

# -------------------------------------------------------------------
# Union-Find
class UnionFind:
    def __init__(self):
        self.parent: Dict[EClassId, EClassId] = {}
        self.rank: Dict[EClassId, int] = {}

    def make(self, x: EClassId) -> None:
        self.parent[x] = x
        self.rank[x] = 0

    def find(self, x: EClassId) -> EClassId:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: EClassId, b: EClassId) -> EClassId:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        # union by rank
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra

# -------------------------------------------------------------------
# E-Graph internals
@dataclass
class ENode:
    op: Op
    children: Tuple[EClassId, ...]   # always canonical leaders

@dataclass
class EClass:
    nodes: List[ENode]
    parents: Set[Tuple[ENodeKey, EClassId]]  # (parent key, parent class)

class EGraph:
    def __init__(self):
        self.uf = UnionFind()
        self.classes: Dict[EClassId, EClass] = {}
        self.hashcons: Dict[ENodeKey, EClassId] = {}
        self.next_id: int = 0
        self.worklist: List[EClassId] = []

    # --------- core helpers ---------
    def fresh_class(self) -> EClassId:
        cid = self.next_id
        self.next_id += 1
        self.uf.make(cid)
        self.classes[cid] = EClass(nodes=[], parents=set())
        return cid

    def canonicalize_children(self, child_ids: Iterable[EClassId]) -> Tuple[EClassId, ...]:
        return tuple(self.uf.find(c) for c in child_ids)

    def add_enode(self, op: Op, child_ids: Tuple[EClassId, ...]) -> EClassId:
        # Numbers: store payload directly; never touch union-find on raw ints
        if op == "Num":
            assert len(child_ids) == 1, "Num expects one payload"
            key = ("Num", (child_ids[0],))   # payload is a plain int
            if key in self.hashcons:
                return self.uf.find(self.hashcons[key])
            cid = self.fresh_class()
            # store as an ENode with the payload in children[0]
            self.classes[cid].nodes.append(ENode("Num", (child_ids[0],)))
            self.hashcons[key] = cid
            return cid

        # Non-literals: canonicalize child e-class IDs and hashcons
        can_children = self.canonicalize_children(child_ids)
        key: ENodeKey = (op, can_children)
        if key in self.hashcons:
            return self.uf.find(self.hashcons[key])

        cid = self.fresh_class()
        self.classes[cid].nodes.append(ENode(op, can_children))
        self.hashcons[key] = cid

        # Add parent pointers from children (only for non-Num nodes)
        for ch in can_children:
            ch = self.uf.find(ch)
            if ch in self.classes:
                self.classes[ch].parents.add((key, cid))
        return cid

    def add_term(self, t: Term) -> EClassId:
        op = t[0]
        if op == "Num":
            # store literal as childless node using payload in children tuple
            return self.add_enode("Num", (t[1],))
        elif op in ("Add", "Mul"):
            a = self.add_term(t[1])
            b = self.add_term(t[2])
            return self.add_enode(op, (a, b))
        else:
            raise ValueError(f"unknown op: {op}")

    # --------- merging & rebuilding ---------
    def merge(self, a: EClassId, b: EClassId) -> EClassId:
        ra, rb = self.uf.find(a), self.uf.find(b)
        if ra == rb:
            return ra
        leader = self.uf.union(ra, rb)
        follower = rb if leader == ra else ra
        if follower in self.classes:
            self.classes[leader].nodes.extend(self.classes[follower].nodes)
            self.classes[leader].parents |= self.classes[follower].parents
            del self.classes[follower]
        self.worklist.append(leader)
        return leader

    def rebuild(self) -> None:
        while self.worklist:
            todo = self.worklist
            self.worklist = []
            # dedup leaders
            uniq: List[EClassId] = []
            seen: Set[EClassId] = set()
            for c in todo:
                rc = self.uf.find(c)
                if rc in self.classes and rc not in seen:
                    uniq.append(rc)
                    seen.add(rc)
            for cid in uniq:
                self._repair(cid)

    def _repair(self, cid: EClassId) -> None:
        # rehash parents of this class; merge congruent parents
        buckets: Dict[ENodeKey, List[EClassId]] = {}
        for (p_key, p_cid) in list(self.classes[cid].parents):
            op, chs = p_key
            can_chs = self.canonicalize_children(chs)
            can_key: ENodeKey = (op, can_chs)
            if can_key in self.hashcons:
                existing = self.uf.find(self.hashcons[can_key])
                pleader = self.uf.find(p_cid)
                if existing != pleader:
                    leader = self.merge(existing, pleader)
                    buckets.setdefault(can_key, []).append(leader)
                else:
                    buckets.setdefault(can_key, []).append(pleader)
            else:
                pleader = self.uf.find(p_cid)
                self.hashcons[can_key] = pleader
                buckets.setdefault(can_key, []).append(pleader)
        # ensure identical parents share one e-class
        for key, pcs in buckets.items():
            base: Optional[EClassId] = None
            for pc in pcs:
                pc = self.uf.find(pc)
                if base is None:
                    base = pc
                elif pc != base:
                    base = self.merge(base, pc)
            # refresh parent back-edges with canonical key
            op, can_chs = key
            for ch in can_chs:
                ch = self.uf.find(ch)
                if ch in self.classes:
                    self.classes[ch].parents.add((key, base))

    # --------- e-matching / rewriting ---------
    def ematch(self, pattern: Term) -> List[Tuple[EClassId, Dict[str, EClassId]]]:
        results: List[Tuple[EClassId, Dict[str, EClassId]]] = []

        def match_in_class(cid: EClassId, pat: Term, subst: Dict[str, EClassId]) -> bool:
            head = pat[0]
            # variable at root of pattern: bind to e-class
            if is_var(head):
                v = head
                root = self.uf.find(cid)
                if v in subst and subst[v] != root:
                    return False
                subst[v] = root
                return True

            # constructor: try nodes in this class with same op
            for node in self.classes[cid].nodes:
                if node.op != head:
                    continue
                saved = dict(subst)
                ok = True
                # handle "Num" specially: payload is in children[0] (a literal)
                if head == "Num":
                    # pattern must be ("Num", ?v) where child is a var
                    if len(pat) != 2 or not is_var(pat[1]):
                        ok = False
                    else:
                        v = pat[1]
                        lit_payload = node.children[0]  # literal value
                        # bind v to the e-class containing this exact literal node
                        # we already are in that class (cid)
                        if v in subst and subst[v] != self.uf.find(cid):
                            ok = False
                        else:
                            subst[v] = self.uf.find(cid)
                else:
                    # non-literal binary ops
                    for child_cid, child_pat in zip(node.children, pat[1:]):
                        if isinstance(child_pat, tuple):
                            if not match_in_class(child_cid, child_pat, subst):
                                ok = False; break
                        else:
                            if not is_var(child_pat):
                                ok = False; break
                            v = child_pat
                            child_leader = self.uf.find(child_cid)
                            if v in subst and subst[v] != child_leader:
                                ok = False; break
                            subst[v] = child_leader
                if ok:
                    return True
                subst.clear(); subst.update(saved)
            return False

        for cid in list(self.classes.keys()):
            if cid not in self.classes:  # may be merged away
                continue
            subst: Dict[str, EClassId] = {}
            if match_in_class(cid, pattern, subst):
                results.append((self.uf.find(cid), subst))
        return results

    def apply(self, match: Tuple[EClassId, Dict[str, EClassId]], rhs: Term) -> bool:
        root, subst = match

        def build(t: Term) -> EClassId:
            head = t[0]
            if is_var(head):
                return subst[head]
            if head == "Num":
                return self.add_enode("Num", (t[1],))
            # binary ops
            kids: List[EClassId] = []
            for ch in t[1:]:
                if isinstance(ch, tuple):
                    kids.append(build(ch))
                else:
                    kids.append(subst[ch])
            return self.add_enode(head, tuple(kids))

        new_c = build(rhs)
        before = (self.uf.find(root), self.uf.find(new_c))
        self.merge(root, new_c)
        after = (self.uf.find(root), self.uf.find(new_c))
        return before != after

    # --------- extraction (size-based) ---------
    def extract(self, cid: EClassId) -> Term:
        best_term: Dict[EClassId, Term] = {}
        best_size: Dict[EClassId, int] = {}

        def size(t: Term) -> int:
            return 1 if t[0] == "Num" else 1 + sum(size(x) for x in t[1:])

        changed = True
        while changed:
            changed = False
            for c in list(self.classes.keys()):
                if c not in self.classes:
                    continue
                best_here: Optional[Tuple[int, Term]] = None
                for node in self.classes[c].nodes:
                    if node.op == "Num":
                        term = ("Num", node.children[0])
                        cand = (1, term)
                    else:
                        # need child bests
                        kids: List[Term] = []
                        ok = True
                        for ch in node.children:
                            ch = self.uf.find(ch)
                            if ch in best_term:
                                kids.append(best_term[ch])
                            else:
                                ok = False; break
                        if not ok:
                            continue
                        term = (node.op, *kids)
                        cand = (size(term), term)
                    if best_here is None or cand[0] < best_here[0]:
                        best_here = cand
                if best_here:
                    prev = best_size.get(c, 1 << 30)
                    if best_here[0] < prev:
                        best_size[c] = best_here[0]
                        best_term[c] = best_here[1]
                        changed = True
        return best_term[self.uf.find(cid)]

# -------------------------------------------------------------------
# Rewrites: (LHS, RHS) patterns with vars like '?a'
COMM_ADD = (("Add", "?a", "?b"), ("Add", "?b", "?a"))
ASSOC_ADD = (("Add", "?a", ("Add", "?b", "?c")), ("Add", ("Add", "?a", "?b"), "?c"))
COMM_MUL = (("Mul", "?a", "?b"), ("Mul", "?b", "?a"))
ASSOC_MUL = (("Mul", "?a", ("Mul", "?b", "?c")), ("Mul", ("Mul", "?a", "?b"), "?c"))
DISTR_L   = (("Mul", "?a", ("Add", "?b", "?c")), ("Add", ("Mul", "?a", "?b"), ("Mul", "?a", "?c")))
DISTR_R   = (("Mul", ("Add", "?a", "?b"), "?c"), ("Add", ("Mul", "?a", "?c"), ("Mul", "?b", "?c")))
REWRITES = [COMM_ADD, ASSOC_ADD, COMM_MUL, ASSOC_MUL, DISTR_L, DISTR_R]

# -------------------------------------------------------------------
# Runner
class Runner:
    def __init__(self, eg: EGraph):
        self.eg = eg

    def run(self, iters: int, rewrites=REWRITES) -> None:
        for _ in range(iters):
            matches: List[Tuple[Tuple[Term, Term], Tuple[EClassId, Dict[str, EClassId]]]] = []
            for lhs, rhs in rewrites:
                for m in self.eg.ematch(lhs):
                    matches.append(((lhs, rhs), m))
            if not matches:
                break
            changed = False
            for (_, rhs), m in matches:
                changed |= self.eg.apply(m, rhs)
            self.eg.rebuild()
            if not changed:
                break

# -------------------------------------------------------------------
# Tests
class TestMiniEgg(unittest.TestCase):
    def test_build_and_extract_identity(self):
        """Test that we can build an expression and extract it back."""
        eg = EGraph()
        expr = ("Mul", ("Add", ("Num", 1), ("Num", 2)),
                      ("Add", ("Num", 3), ("Add", ("Num", 4), ("Num", 5))))
        root = eg.add_term(expr)
        Runner(eg).run(3, rewrites=REWRITES)
        best = eg.extract(root)
        # With only comm/assoc/distrib, best size typically stays similar to input
        self.assertEqual(best[0], "Mul")
        self.assertEqual(best[1][0], "Add")
        self.assertEqual(best[2][0], "Add")

    def test_commutativity_add(self):
        """Test that addition is commutative: a + b = b + a."""
        eg = EGraph()
        a = eg.add_term(("Add", ("Num", 1), ("Num", 2)))
        b = eg.add_term(("Add", ("Num", 2), ("Num", 1)))
        # before rewrites, they are different classes
        self.assertNotEqual(eg.uf.find(a), eg.uf.find(b))
        Runner(eg).run(2, rewrites=[COMM_ADD])  # only comm add
        # after, they should be equal
        self.assertEqual(eg.uf.find(a), eg.uf.find(b))

    def test_commutativity_mul(self):
        """Test that multiplication is commutative: a * b = b * a."""
        eg = EGraph()
        a = eg.add_term(("Mul", ("Num", 3), ("Num", 4)))
        b = eg.add_term(("Mul", ("Num", 4), ("Num", 3)))
        # before rewrites, they are different classes
        self.assertNotEqual(eg.uf.find(a), eg.uf.find(b))
        Runner(eg).run(2, rewrites=[COMM_MUL])
        # after, they should be equal
        self.assertEqual(eg.uf.find(a), eg.uf.find(b))

    def test_associativity_add(self):
        """Test that addition is associative: (a + b) + c = a + (b + c)."""
        eg = EGraph()
        left = eg.add_term(("Add", ("Add", ("Num", 1), ("Num", 2)), ("Num", 3)))
        right = eg.add_term(("Add", ("Num", 1), ("Add", ("Num", 2), ("Num", 3))))
        # before rewrites, they are different classes
        self.assertNotEqual(eg.uf.find(left), eg.uf.find(right))
        Runner(eg).run(3, rewrites=[ASSOC_ADD])
        # after, they should be equal
        self.assertEqual(eg.uf.find(left), eg.uf.find(right))

    def test_associativity_mul(self):
        """Test that multiplication is associative: (a * b) * c = a * (b * c)."""
        eg = EGraph()
        left = eg.add_term(("Mul", ("Mul", ("Num", 2), ("Num", 3)), ("Num", 4)))
        right = eg.add_term(("Mul", ("Num", 2), ("Mul", ("Num", 3), ("Num", 4))))
        # before rewrites, they are different classes
        self.assertNotEqual(eg.uf.find(left), eg.uf.find(right))
        Runner(eg).run(3, rewrites=[ASSOC_MUL])
        # after, they should be equal
        self.assertEqual(eg.uf.find(left), eg.uf.find(right))

    def test_distributivity_left(self):
        """Test left distributivity: a * (b + c) = (a * b) + (a * c)."""
        eg = EGraph()
        left = eg.add_term(("Mul", ("Num", 5), ("Add", ("Num", 2), ("Num", 3))))
        right = eg.add_term(("Add", ("Mul", ("Num", 5), ("Num", 2)),
                                   ("Mul", ("Num", 5), ("Num", 3))))
        # before rewrites, they are different classes
        self.assertNotEqual(eg.uf.find(left), eg.uf.find(right))
        Runner(eg).run(3, rewrites=[DISTR_L])
        # after, they should be equal
        self.assertEqual(eg.uf.find(left), eg.uf.find(right))

    def test_distributivity_right(self):
        """Test right distributivity: (a + b) * c = (a * c) + (b * c)."""
        eg = EGraph()
        left = eg.add_term(("Mul", ("Add", ("Num", 1), ("Num", 2)), ("Num", 4)))
        right = eg.add_term(("Add", ("Mul", ("Num", 1), ("Num", 4)),
                                   ("Mul", ("Num", 2), ("Num", 4))))
        # before rewrites, they are different classes
        self.assertNotEqual(eg.uf.find(left), eg.uf.find(right))
        Runner(eg).run(3, rewrites=[DISTR_R])
        # after, they should be equal
        self.assertEqual(eg.uf.find(left), eg.uf.find(right))

    def test_distributivity_creates_equal_form(self):
        """Test that distributivity can prove complex equalities."""
        eg = EGraph()
        x = eg.add_term(("Mul", ("Num", 7), ("Add", ("Num", 2), ("Num", 3))))
        y = eg.add_term(("Add", ("Mul", ("Num", 7), ("Num", 2)),
                               ("Mul", ("Num", 7), ("Num", 3))))
        # prove equality via distributivity rewrites
        Runner(eg).run(4, rewrites=[DISTR_L, DISTR_R])
        self.assertEqual(eg.uf.find(x), eg.uf.find(y))

    def test_assoc_then_comm(self):
        """Test that associativity and commutativity work together."""
        eg = EGraph()
        t1 = eg.add_term(("Add", ("Num", 1), ("Add", ("Num", 2), ("Num", 3))))
        t2 = eg.add_term(("Add", ("Add", ("Num", 3), ("Num", 2)), ("Num", 1)))
        Runner(eg).run(5, rewrites=[ASSOC_ADD, COMM_ADD])
        self.assertEqual(eg.uf.find(t1), eg.uf.find(t2))

    def test_complex_equality_proof(self):
        """Test a complex equality that requires multiple rewrite rules."""
        eg = EGraph()
        # (a + b) * (c + d) should equal (a * c) + (a * d) + (b * c) + (b * d)
        left = eg.add_term(("Mul", ("Add", ("Num", 1), ("Num", 2)),
                                  ("Add", ("Num", 3), ("Num", 4))))
        right = eg.add_term(("Add", ("Add", ("Mul", ("Num", 1), ("Num", 3)),
                                          ("Mul", ("Num", 1), ("Num", 4))),
                                  ("Add", ("Mul", ("Num", 2), ("Num", 3)),
                                          ("Mul", ("Num", 2), ("Num", 4)))))
        # Apply all rewrite rules
        Runner(eg).run(10, rewrites=REWRITES)
        self.assertEqual(eg.uf.find(left), eg.uf.find(right))

    def test_pattern_matching(self):
        """Test that pattern matching works correctly."""
        eg = EGraph()
        # Add some expressions
        eg.add_term(("Add", ("Num", 1), ("Num", 2)))
        eg.add_term(("Add", ("Num", 3), ("Num", 4)))
        eg.add_term(("Mul", ("Num", 5), ("Num", 6)))
        
        # Find all additions
        pattern = ("Add", "?x", "?y")
        matches = eg.ematch(pattern)
        self.assertEqual(len(matches), 2)  # Should find 2 additions
        
        # Check that we found the right patterns
        found_ops = set()
        for _, subst in matches:
            x_class = subst['?x']
            y_class = subst['?y']
            # Extract the actual numbers
            x_val = eg.extract(x_class)[1]  # ("Num", value)
            y_val = eg.extract(y_class)[1]
            found_ops.add((x_val, y_val))
        
        expected = {(1, 2), (3, 4)}
        self.assertEqual(found_ops, expected)

    def test_extraction_optimization(self):
        """Test that extraction finds smaller representations."""
        eg = EGraph()
        # Start with a complex expression
        expr = ("Add", ("Mul", ("Num", 0), ("Add", ("Num", 1), ("Num", 2))),
                      ("Mul", ("Num", 1), ("Num", 0)))
        root = eg.add_term(expr)
        
        # Apply rewrite rules
        Runner(eg).run(5, rewrites=REWRITES)
        
        # Extract should find a simpler form
        optimized = eg.extract(root)
        # The result should be simpler (0 * anything = 0, so this should be 0)
        self.assertEqual(optimized, ("Num", 0))

    def test_union_find_operations(self):
        """Test that union-find operations work correctly."""
        eg = EGraph()
        
        # Add some terms
        a = eg.add_term(("Num", 1))
        b = eg.add_term(("Num", 2))
        c = eg.add_term(("Num", 1))  # Same as a
        
        # Initially different
        self.assertNotEqual(eg.uf.find(a), eg.uf.find(b))
        self.assertNotEqual(eg.uf.find(a), eg.uf.find(c))
        
        # Merge a and c (they represent the same number)
        eg.merge(a, c)
        self.assertEqual(eg.uf.find(a), eg.uf.find(c))
        self.assertNotEqual(eg.uf.find(a), eg.uf.find(b))

    def test_rebuild_invariants(self):
        """Test that rebuilding maintains e-graph invariants."""
        eg = EGraph()
        
        # Add a complex expression
        expr = ("Add", ("Add", ("Num", 1), ("Num", 2)), ("Num", 3))
        root = eg.add_term(expr)
        
        # Apply some rewrites to trigger rebuilding
        Runner(eg).run(3, rewrites=[COMM_ADD, ASSOC_ADD])
        
        # The e-graph should still be valid
        # (This is more of a smoke test - the real test is that it doesn't crash)
        optimized = eg.extract(root)
        self.assertIsInstance(optimized, tuple)
        self.assertGreater(len(optimized), 0)

    def test_empty_egraph(self):
        """Test operations on an empty e-graph."""
        eg = EGraph()
        
        # Pattern matching on empty graph should return empty list
        matches = eg.ematch(("Add", "?x", "?y"))
        self.assertEqual(len(matches), 0)
        
        # Extract should handle non-existent class gracefully
        with self.assertRaises(KeyError):
            eg.extract(999)  # Non-existent class ID

    def test_single_number(self):
        """Test that single numbers work correctly."""
        eg = EGraph()
        
        # Add a single number
        num = eg.add_term(("Num", 42))
        
        # Extract should return the same number
        extracted = eg.extract(num)
        self.assertEqual(extracted, ("Num", 42))
        
        # Pattern matching should work
        pattern = ("Num", "?x")
        matches = eg.ematch(pattern)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][1]['?x'], eg.uf.find(num))

if __name__ == "__main__":
    unittest.main(verbosity=2)