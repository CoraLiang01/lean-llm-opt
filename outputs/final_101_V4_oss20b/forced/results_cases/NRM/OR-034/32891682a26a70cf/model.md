##### Abstract Model

**Index Set**  
\(I = \{i \mid \text{product } i \text{ in the bakery dataset}\}\)

**Parameters**  
\(w_i =\) revenue per unit of product \(i\)  
\(d_i =\) deterministic demand for product \(i\) over the sales horizon  
\(c_i =\) initial inventory of product \(i\)

**Decision Variables**  
\(x_i =\) quantity of product \(i\) the bakery plans to fulfill  
\(x_i \in \mathbb{Z}_{+}\) for all \(i \in I\)

**Objective Function**  
\[
\max \sum_{i \in I} w_i \, x_i
\]

**Constraints**  

1. **Inventory Constraints**  
\[
x_i \leq c_i \quad \forall i \in I
\]

2. **Demand Constraints**  
\[
x_i \leq d_i \quad \forall i \in I
\]

**Data Mapping**  

| Parameter | Source Table | Column Name |
|-----------|--------------|-------------|
| \(w_i\)   | file_0_view_0 | Revenue |
| \(d_i\)   | file_0_view_0 | Demand |
| \(c_i\)   | file_0_view_0 | Initial Inventory |