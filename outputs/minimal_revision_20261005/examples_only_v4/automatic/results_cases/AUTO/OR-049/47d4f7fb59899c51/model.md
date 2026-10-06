ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of shelves, indexed by 𝑖. (ShelfID from file_0_view_0)
- 𝑃: Set of products, indexed by 𝑗. (ProductName from file_1_view_0)

Parameters:
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦_𝑖: Capacity of shelf 𝑖. (file_0_view_0, column: Capacity)
- 𝑉𝑎𝑙𝑢𝑒_𝑗: Value per unit of product 𝑗. (file_1_view_0, column: Value)
- 𝑊𝑒𝑖𝑔ℎ𝑡_𝑗: Weight per unit of product 𝑗. (file_1_view_0, column: Weight)

Decision Variables:
- 𝑥_{𝑖𝑗}: Number of units of product 𝑗 to place on shelf 𝑖. (integer, 𝑥_{𝑖𝑗} ≥ 0)

Objective:
Maximize total value of products displayed:
\[
\max \sum_{i \in S} \sum_{j \in P} Value_j \cdot x_{ij}
\]

Constraints:
1. Shelf capacity constraints (for each shelf 𝑖 ∈ 𝑆):
\[
\sum_{j \in P} Weight_j \cdot x_{ij} \leq Capacity_i \qquad \forall i \in S
\]
2. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
\]

DATA MAPPING

- 𝑆 (Shelves): file_0_view_0, column: ShelfID
- 𝑃 (Products): file_1_view_0, column: ProductName
- 𝐶𝑎𝑝𝑎𝑐𝑖𝑡𝑦_𝑖: file_0_view_0, columns: ShelfID, Capacity (map ShelfID to Capacity)
- 𝑉𝑎𝑙𝑢𝑒_𝑗: file_1_view_0, columns: ProductName, Value (map ProductName to Value)
- 𝑊𝑒𝑖𝑔ℎ𝑡_𝑗: file_1_view_0, columns: ProductName, Weight (map ProductName to Weight)
- 𝑥_{𝑖𝑗}: Decision variable for each (ShelfID, ProductName) pair

All index sets, parameters, and constraints are mapped directly from the supplied data. No data is omitted or synthesized.