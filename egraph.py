# mini_egraph_fixed.py — minimal e-graph with deferred rebuild

class UnionFind:
    def __init__(self):
        self.parent = {}

    def make(self, x):
        self.parent[x] = x

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
        return self.find(a)


class EGraph:
    def __init__(self):
        self.uf = UnionFind()
        self.classes = {}            # cid -> list of enode keys
        self.hashcons = {}           # enode key -> cid
        self.next_id = 0
        self.worklist = []

    def fresh_class(self):
        cid = self.next_id
        self.next_id += 1
        self.uf.make(cid)
        self.classes[cid] = []
        return cid

    def add_enode(self, key):
        """
        key is a canonical enode key:
          - for numbers: ("Num", n)
          - for ops:     ("Add", (cidA, cidB)) or ("Mul", (cidA, cidB))
        """
        if key in self.hashcons:
            return self.uf.find(self.hashcons[key])
        cid = self.fresh_class()
        self.classes[cid].append(key)
        self.hashcons[key] = cid
        return cid

    def add_term(self, term):
        """
        term is either:
          - int (a literal)
          - tuple: ("Add"|"Mul", left_term, right_term)
        """
        if isinstance(term, int):
            return self.add_enode(("Num", term))  # no children
        op, a, b = term
        ca = self.add_term(a)
        cb = self.add_term(b)
        # canonicalize child class ids
        ca, cb = self.uf.find(ca), self.uf.find(cb)
        if op not in ("Add", "Mul"):
            raise ValueError("unknown op")
        return self.add_enode((op, (ca, cb)))

    def merge(self, a, b):
        ra, rb = self.uf.find(a), self.uf.find(b)
        if ra == rb:
            return ra
        leader = self.uf.union(ra, rb)
        self.worklist.append(leader)
        return leader

    def rebuild(self):
        # Process pending merges; rehash parents to dedup congruent enodes.
        while self.worklist:
            todo = self.worklist
            self.worklist = []
            # rehash every enode in these classes
            for cid in list(todo):
                cid = self.uf.find(cid)
                if cid not in self.classes:
                    continue
                new_nodes = []
                for key in self.classes[cid]:
                    op, payload = key
                    if op == "Num":
                        can_key = key  # no children
                    else:
                        a, b = payload
                        can_key = (op, (self.uf.find(a), self.uf.find(b)))
                    # dedup via hashcons
                    if can_key in self.hashcons:
                        other = self.uf.find(self.hashcons[can_key])
                        if other != cid:
                            self.merge(cid, other)
                    else:
                        self.hashcons[can_key] = cid
                        new_nodes.append(can_key)
                self.classes[cid] = new_nodes


# ---- Test case ----
if __name__ == "__main__":
    eg = EGraph()
    expr = ("Mul", ("Add", 1, 2), ("Add", 3, 4))
    root = eg.add_term(expr)

    # simple rewrite: Add(a,b) -> Add(b,a)
    for cid, nodes in list(eg.classes.items()):
        for key in nodes:
            op, payload = key
            if op == "Add":
                a, b = payload
                rhs = eg.add_enode(("Add", (eg.uf.find(b), eg.uf.find(a))))
                eg.merge(cid, rhs)

    eg.rebuild()

    print("Root e-class:", eg.uf.find(root))
    print("#classes:", len({eg.uf.find(c) for c in eg.classes}))
    print("Enodes by class:")
    for c in sorted({eg.uf.find(cid) for cid in eg.classes}):
        print(" ", c, "->", eg.classes[c])