Mathematical Model (Abstract Formulation)

Index Sets:
- \( P \): Set of products, indexed by \( p \) (corresponds to all "Product Name" in file_0_view_0).

Parameters:
- \( a_p \): Labor required per unit of product \( p \) (from "Labor per unit" in file_0_view_0).
- \( b_p \): Material required per unit of product \( p \) (from "Material per unit" in file_0_view_0).
- \( s_p \): Selling price per unit of product \( p \) (from "Selling Price" in file_0_view_0).
- \( v_p \): Variable cost per unit of product \( p \) (from "Variable Cost" in file_0_view_0).
- \( L \): Total weekly labor capacity (given as 1,650).
- \( M \): Total weekly material capacity (given as 1,850).
- \( F \): Fixed weekly operating cost (given as 4,500).

Decision Variables:
- \( x_p \geq 0 \): Continuous production quantity of product \( p \) (units).

Objective:
\[
\max_{x_p \geq 0} \left[ \sum_{p \in P} (s_p - v_p) x_p - F \right]
\]

Subject to:
\[
\sum_{p \in P} a_p x_p \leq L \qquad \text{(Labor capacity constraint)}
\]
\[
\sum_{p \in P} b_p x_p \leq M \qquad \text{(Material capacity constraint)}
\]
\[
x_p \geq 0 \qquad \forall p \in P
\]

Data Mapping

- Index set \( P \): All rows in file_0_view_0, identified by "Product Name".
- \( a_p \): file_0_view_0, column "Labor per unit", keyed by "Product Name".
- \( b_p \): file_0_view_0, column "Material per unit", keyed by "Product Name".
- \( s_p \): file_0_view_0, column "Selling Price", keyed by "Product Name".
- \( v_p \): file_0_view_0, column "Variable Cost", keyed by "Product Name".
- \( L \): 1,650 (from user description).
- \( M \): 1,850 (from user description).
- \( F \): 4,500 (from user description).
- Decision variable \( x_p \): Nonnegative continuous, indexed by "Product Name" in file_0_view_0.