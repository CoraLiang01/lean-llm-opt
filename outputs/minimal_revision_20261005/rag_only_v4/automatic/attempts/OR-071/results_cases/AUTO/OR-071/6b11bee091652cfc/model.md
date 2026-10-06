ABSTRACT MATHEMATICAL MODEL

Index Sets:
- \( P \): Set of products, indexed by \( p \) (corresponding to each "Product Name" in 41.csv).

Parameters:
- \( a_p \): Labor required per unit of product \( p \). [From "Labor per unit" in file_0_view_0]
- \( b_p \): Material required per unit of product \( p \). [From "Material per unit" in file_0_view_0]
- \( s_p \): Selling price per unit of product \( p \). [From "Selling Price" in file_0_view_0]
- \( v_p \): Variable cost per unit of product \( p \). [From "Variable Cost" in file_0_view_0]
- \( L \): Total weekly labor capacity (= 1,650 units). [From user query]
- \( M \): Total weekly material capacity (= 1,850 units). [From user query]
- \( F \): Fixed weekly operating cost (= \$4,500). [From user query]

Decision Variables:
- \( x_p \geq 0 \): Continuous quantity of product \( p \) to produce (units).

Objective:
\[
\max_{x_p \geq 0} \left\{ \sum_{p \in P} (s_p - v_p) x_p - F \right\}
\]
(Maximize weekly net profit: total sales revenue minus total variable cost minus fixed cost.)

Constraints:
\[
\sum_{p \in P} a_p x_p \leq L
\]
(Total labor used does not exceed labor capacity.)

\[
\sum_{p \in P} b_p x_p \leq M
\]
(Total material used does not exceed material capacity.)

\[
x_p \geq 0 \quad \forall p \in P
\]
(Nonnegativity and continuity of production quantities.)

---

DATA MAPPING

- Index set \( P \): All rows in table_id file_0_view_0, column "Product Name".
- Parameter \( a_p \): file_0_view_0, column "Labor per unit", keyed by "Product Name".
- Parameter \( b_p \): file_0_view_0, column "Material per unit", keyed by "Product Name".
- Parameter \( s_p \): file_0_view_0, column "Selling Price", keyed by "Product Name".
- Parameter \( v_p \): file_0_view_0, column "Variable Cost", keyed by "Product Name".
- Parameters \( L = 1650 \), \( M = 1850 \), \( F = 4500 \): from user query.