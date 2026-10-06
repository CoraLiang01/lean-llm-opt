ABSTRACT MATHEMATICAL MODEL

Index Sets:
- \( P \): Set of products, indexed by \( p \). Each product is uniquely identified by its "Product Name" (see Data Mapping).

Parameters:
- \( a_p \): Labor required per unit of product \( p \).
- \( b_p \): Material required per unit of product \( p \).
- \( s_p \): Selling price per unit of product \( p \).
- \( v_p \): Variable cost per unit of product \( p \).
- \( L \): Total weekly labor capacity (1,650 units).
- \( M \): Total weekly material capacity (1,850 units).
- \( F \): Fixed weekly operating cost (\$4,500).

Variables:
- \( x_p \geq 0 \): Continuous quantity of product \( p \) to produce (may be fractional).

Objective:
\[
\max \left\{ \sum_{p \in P} (s_p - v_p) x_p - F \right\}
\]
That is, maximize total sales revenue minus total variable cost minus fixed weekly operating cost.

Constraints:
1. Labor capacity:
\[
\sum_{p \in P} a_p x_p \leq L
\]
2. Material capacity:
\[
\sum_{p \in P} b_p x_p \leq M
\]
3. Nonnegativity:
\[
x_p \geq 0 \quad \forall p \in P
\]

Data Mapping:

- Index set \( P \) and all parameters are mapped as follows (from CSVQA_DATA table_id = "file_0_view_0"):
    - \( p \): "Product Name"
    - \( a_p \): "Labor per unit"
    - \( b_p \): "Material per unit"
    - \( s_p \): "Selling Price"
    - \( v_p \): "Variable Cost"
- \( L = 1650 \), \( M = 1850 \), \( F = 4500 \) (from user description).

All 198 products from "file_0_view_0" are included, with their coefficients as given in the respective columns. No products or constraints are omitted.