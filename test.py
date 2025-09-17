from mini_egg import EGraph, Runner, COMM_ADD, ASSOC_ADD, REWRITES

print('--------------------------------')
print("Test: Commutativity")
eg1 = EGraph()

expr1 = ("Add", ("Num", 1), ("Num", 2))  # 1 + 2
expr2 = ("Add", ("Num", 2), ("Num", 1))  # 2 + 1
# should be different before COMM_ADD
root1 = eg1.add_term(expr1)
root2 = eg1.add_term(expr2)

print(f"Before COMM_ADD:")
print(f"  root1 class: {eg1.uf.find(root1)}")
print(f"  root2 class: {eg1.uf.find(root2)}")
print(f"  Equal? {eg1.uf.find(root1) == eg1.uf.find(root2)}")

# Apply only COMM_ADD, should be equal after
Runner(eg1).run(iters=3, rewrites=[COMM_ADD])

print(f"After COMM_ADD:")
print(f"  root1 class: {eg1.uf.find(root1)}")
print(f"  root2 class: {eg1.uf.find(root2)}")
print(f"  Equal? {eg1.uf.find(root1) == eg1.uf.find(root2)}")
print('--------------------------------')