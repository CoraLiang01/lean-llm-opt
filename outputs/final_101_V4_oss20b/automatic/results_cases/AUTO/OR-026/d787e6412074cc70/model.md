**Abstract Optimization Model**

**Index Set**  
\(I\) – set of all fashion products in the dataset.

**Parameters**  
\[
\begin{aligned}
w_i &\quad\text{revenue per unit of product }i, && i\in I\\
d_i &\quad\text{deterministic demand for product }i, && i\in I\\
c_i &\quad\text{initial inventory of product }i, && i\in I
\end{aligned}
\]

**Decision Variables**  
\[
x_i \in \mathbb{Z}_{>0}\quad\text{units of product }i\text{ to fulfill},\qquad i\in I
\]

**Objective Function**  
\[
\max \;\sum_{i\in I} w_i\,x_i
\]

**Constraints**

1. **Inventory Constraints**  
\[
x_i \leq c_i,\qquad \forall i\in I
\]

2. **Demand Constraints**  
\[
x_i \leq d_i,\qquad \forall i\in I
\]

**Data Mapping**

| Source | Table ID | Column Name | Parameter |
|--------|----------|-------------|-----------|
| SupermarketSales.csv | file_0_view_0 | Product Name | product identifier (implicit in \(I\)) |
| SupermarketSales.csv | file_0_view_0 | Revenue | \(w_i\) |
| SupermarketSales.csv | file_0_view_0 | Demand | \(d_i\) |
| SupermarketSales.csv | file_0_view_0 | Initial Inventory | \(c_i\) |

*The filter “Category contains 'Fashion'” was applied when constructing the index set \(I\). No additional data values or record counts are included.*