# mini_egraph.py -- a tiny e-graph with rebuilding

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
        self.classes = {}
        self.hashcons = {}
        self.next_id = 0
        self.worklist = []

    def fresh_class(self):
        cid = self.next_id
        self.next_id += 1
        self.uf.make(cid)
        self.classes[cid] = []
        return cid

    def add_enode(self, op, children):
        key = (op, tuple(self.uf.find(c) for c in children))
        if key in self.hashcons:
            return self.uf.find(self.hashcons[key])
        cid = self.fresh_class()
        self.classes[cid].append(key)
        self.hashcons[key] = cid
        return cid

    def add_term(self, term):
        if isinstance(term, int):
            return self.add_enode("Num", (term,))
        op, a, b = term
        ca = self.add_term(a)
        cb = self.add_term(b)
        return self.add_enode(op, (ca, cb))

    def merge(self, a, b):
        ra, rb = self.uf.find(a), self.uf.find(b)
        if ra == rb:
            return ra
        leader = self.uf.union(ra, rb)
        self.worklist.append(leader)
        return leader

    def rebuild(self):
        while self.worklist:
            todo = self.worklist
            self.worklist = []
            for cid in todo:
                # re-canonicalize nodes in this class
                new_nodes = []
                for (op, chs) in self.classes.get(cid, []):
                    can_chs = tuple(self.uf.find(c) for c in chs)
                    key = (op, can_chs)
                    if key in self.hashcons:
                        other = self.uf.find(self.hashcons[key])
                        if other != cid:
                            self.merge(cid, other)
                    else:
                        self.hashcons[key] = cid
                        new_nodes.append((op, can_chs))
                self.classes[cid] = new_nodes


# Simple test
# Builds an e-graph for (* (+ 1 2) (+ 3 4))
eg = EGraph()
expr = ("Mul", ("Add", 1, 2), ("Add", 3, 4))
root = eg.add_term(expr)

# Rewrite: commutativity of Add
for cid, nodes in list(eg.classes.items()):
    for (op, chs) in nodes:
        if op == "Add":
            a, b = chs
            rhs = eg.add_enode("Add", (b, a))
            eg.merge(cid, rhs)

eg.rebuild()

print("Root eclass:", root)
print("Classes:", eg.classes)