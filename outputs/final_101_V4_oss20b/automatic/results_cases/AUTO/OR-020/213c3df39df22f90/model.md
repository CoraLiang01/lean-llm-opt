**Abstract Model**

**Index set**  
\(I\) – set of all products in the supermarket.

**Parameters**  
\[
\begin{aligned}
w_i &\;=\; \text{Revenue per unit of product } i, && i\in I\\
d_i &\;=\; \text{Deterministic demand for product } i, && i\in I\\
c_i &\;=\; \text{Initial inventory of product } i, && i\in I
\end{aligned}
\]

**Decision variables**  
\[
x_i \;\;=\;\; \text{Number of units of product } i \text{ the store plans to fulfill}, \qquad i\in I
\]

**Objective function**  
\[
\max \;\sum_{i\in I} w_i\,x_i
\]

**Constraints**

1. **Demand constraints**  
\[
x_i \;\le\; d_i, \qquad \forall i\in I
\]

2. **Inventory constraints**  
\[
x_i \;\le\; c_i, \qquad \forall i\in I
\]

3. **Variable domain**  
\[
x_i \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall i\in I
\]

---

**Data Mapping**

| Parameter | Source table | Source column |
|-----------|--------------|---------------|
| \(w_i\)   | file_0_view_0 | Revenue |
| \(d_i\)   | file_0_view_0 | Demand |
| \(c_i\)   | file_0_view_0 | Initial Inventory |

The table **file_0_view_0** contains the columns **Product Name**, **Revenue**, **Demand**, and **Initial Inventory** for all 109 products listed in the dataset. Each row corresponds to a distinct product \(i\in I\).