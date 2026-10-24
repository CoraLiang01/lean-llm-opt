**Mathematical Model**

Let  

- \(I\) be the set of vehicle types indexed by the value of **ProductName** in *products.csv*.

Parameters  
\[
\begin{aligned}
\text{Value}_i &\;=\; \text{Value column of product } i \quad (i\in I)\\
\text{Weight}_i &\;=\; \text{Weight column of product } i \quad (i\in I)\\
\text{Capacity} &\;=\; \text{Capacity column of the single row in capacity.csv}
\end{aligned}
\]

Decision variables  
\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad (i\in I)
\]

Objective (maximize total benefit)  
\[
\max \;\sum_{i\in I} \text{Value}_i \; x_i
\]

Inventory capacity constraint  
\[
\sum_{i\in I} \text{Weight}_i \; x_i \;\le\; \text{Capacity}
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| Value\_i  | file_1_view_0 | Value |
| Weight\_i | file_1_view_0 | Weight |
| Capacity  | file_0_view_0 | Capacity |

All index sets, parameters, variables, objective, and constraints are defined symbolically; no numerical values are hard‑coded.