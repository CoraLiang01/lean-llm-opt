ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑆: Set of displays (shelves), indexed by \( i \), with ShelfID from file_0_view_0.
- 𝑃: Set of products, indexed by \( j \), with ProductName from file_1_view_0.

Parameters:
- \( C_i \): Capacity of display \( i \) (from file_0_view_0, column Capacity).
- \( v_j \): Value per unit of product \( j \) (from file_1_view_0, column Value).
- \( w_j \): Weight per unit of product \( j \) (from file_1_view_0, column Weight).

Decision Variables:
- \( x_{ij} \): Number of units of product \( j \) placed on display \( i \). (Domain: nonnegative integers, \( x_{ij} \in \mathbb{Z}_+ \))

Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \, x_{ij}
\]

Subject to:

1. Display Capacity Constraints (for each \( i \in S \)):
\[
\sum_{j \in P} w_j \, x_{ij} \leq C_i
\]

2. Minimum Placement of First Product (let \( j^* \) be the ProductName in the first row of file_1_view_0):
\[
\sum_{i \in S} x_{i j^*} \geq 5
\]

3. Nonnegativity and Integrality:
\[
x_{ij} \in \mathbb{Z}_+, \quad \forall i \in S,\, j \in P
\]

---

DATA MAPPING

- Index set \( S \): All ShelfID values from file_0_view_0, column ShelfID.
- Index set \( P \): All ProductName values from file_1_view_0, column ProductName.
- Parameter \( C_i \): file_0_view_0, column Capacity, keyed by ShelfID.
- Parameter \( v_j \): file_1_view_0, column Value, keyed by ProductName.
- Parameter \( w_j \): file_1_view_0, column Weight, keyed by ProductName.
- The "first product" \( j^* \): ProductName in the first row of file_1_view_0.
- Decision variable \( x_{ij} \): Number of units of product \( j \) placed on display \( i \), indexed by (ShelfID, ProductName).