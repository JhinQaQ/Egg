# Egg Python

A Python implementation of E-graphs (equality graphs) for term rewriting and program optimization, inspired by the [Egg](https://egraphs-good.github.io/) library.

## What are E-graphs?

E-graphs are a data structure that efficiently represents many equivalent expressions simultaneously. They're particularly useful for:

- **Program optimization**: Finding equivalent but more efficient versions of expressions
- **Theorem proving**: Discovering equalities between terms
- **Compiler optimizations**: Applying algebraic rewrites (commutativity, associativity, distributivity)
- **Symbolic computation**: Manipulating mathematical expressions

## Features

- **Union-Find data structure** for efficient equivalence class management
- **Hash-consing** to avoid duplicate nodes
- **Deferred rebuilding** for performance
- **Pattern matching** with variables (e.g., `?x`, `?y`)
- **Term rewriting** with customizable rules
- **Extraction** of optimal terms based on size
- **Comprehensive test suite**

## Project Structure

- `mini_egg.py` - Full-featured implementation with rewriting, pattern matching, and extraction
- `egraph.py` - Minimal implementation focusing on core E-graph operations

## Quick Start

### Basic Usage

```python
from mini_egg import EGraph, Runner, REWRITES

# Create an E-graph
eg = EGraph()

# Add an expression: (1 + 2) * (3 + 4)
expr = ("Mul", ("Add", ("Num", 1), ("Num", 2)),
              ("Add", ("Num", 3), ("Num", 4)))
root = eg.add_term(expr)

# Apply rewrite rules
runner = Runner(eg)
runner.run(iterations=5, rewrites=REWRITES)

# Extract the optimized term
optimized = eg.extract(root)
print(optimized)  # May show a more efficient form
```

### Available Rewrite Rules

The library includes common algebraic rewrite rules:

- **Commutativity**: `Add(a, b) → Add(b, a)`, `Mul(a, b) → Mul(b, a)`
- **Associativity**: `Add(a, Add(b, c)) → Add(Add(a, b), c)`
- **Distributivity**: `Mul(a, Add(b, c)) → Add(Mul(a, b), Mul(a, c))`

### Pattern Matching

```python
# Find all additions in the E-graph
pattern = ("Add", "?x", "?y")
matches = eg.ematch(pattern)
for root_class, substitutions in matches:
    print(f"Found Add({substitutions['?x']}, {substitutions['?y']})")
```

## Examples

### Example 1: Proving Equality

```python
from mini_egg import EGraph, Runner, COMM_ADD, ASSOC_ADD

eg = EGraph()

# Add two expressions that should be equal
expr1 = ("Add", ("Num", 1), ("Add", ("Num", 2), ("Num", 3)))
expr2 = ("Add", ("Add", ("Num", 3), ("Num", 2)), ("Num", 1))

root1 = eg.add_term(expr1)
root2 = eg.add_term(expr2)

# Apply commutativity and associativity
Runner(eg).run(5, rewrites=[COMM_ADD, ASSOC_ADD])

# Check if they're now in the same equivalence class
print(eg.uf.find(root1) == eg.uf.find(root2))  # Should be True
```

### Example 2: Optimization

```python
from mini_egg import EGraph, Runner, REWRITES

eg = EGraph()

# Add expression: 7 * (2 + 3)
expr = ("Mul", ("Num", 7), ("Add", ("Num", 2), ("Num", 3)))
root = eg.add_term(expr)

# Apply all rewrite rules
Runner(eg).run(10, rewrites=REWRITES)

# Extract the best (smallest) representation
best = eg.extract(root)
print(f"Original: {expr}")
print(f"Optimized: {best}")
```

## Running Tests

```bash
python mini_egg.py
```

The test suite includes:
- Basic E-graph operations
- Commutativity and associativity tests
- Distributivity proofs
- Pattern matching verification

## Implementation Details

### Core Components

1. **Union-Find**: Manages equivalence classes efficiently
2. **E-nodes**: Represent individual operations (Add, Mul, Num)
3. **E-classes**: Groups of equivalent E-nodes
4. **Hash-consing**: Prevents duplicate nodes
5. **Rebuilding**: Maintains invariants after merges

### Data Structures

- **Terms**: Tuples like `("Add", ("Num", 1), ("Num", 2))`
- **Pattern Variables**: Strings starting with `?` (e.g., `"?x"`)
- **E-class IDs**: Integers representing equivalence classes

## Notes

- The implementation uses deferred rebuilding for better performance
- Hash-consing prevents exponential blowup in many cases
- Union-Find with path compression ensures near-constant time operations
- Pattern matching is optimized for common cases

## Limitations

This is a minimal implementation focused on clarity and educational value. For production use, consider:

- More sophisticated extraction heuristics
- Additional rewrite rules
- Better memory management
- Parallel processing support

## References

- [Egg: Fast and Extensible Equality Saturation](https://egraphs-good.github.io/)
- [E-graphs and Equality Saturation](https://dl.acm.org/doi/10.1145/3434304)
- [The E-graph Data Structure](https://www.hillelwayne.com/post/equality-saturation/)

