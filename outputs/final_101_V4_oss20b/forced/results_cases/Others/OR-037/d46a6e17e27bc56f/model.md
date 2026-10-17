**Mathematical Model**

Let  

- \(I\) be the set of vehicle types (indexed by the values of *ProductName*).  
- \(v_i\) be the profit per unit of type \(i\) (from *Value*).  
- \(w_i\) be the capacity consumption per unit of type \(i\) (from *Weight*).  
- \(C\) be the total inventory capacity (from *Capacity*).  
- \(x_i\) be the number of vehicles of type \(i\) ordered per day.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{+}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Set | Source | Table ID | Column(s) | Mapping |
|-----------------|--------|----------|-----------|---------|
| \(I\) | products.csv | `file_1_view_0` | `ProductName` | Index set of all 25 product names |
| \(v_i\) | products.csv | `file_1_view_0` | `Value` | Profit per unit for each product |
| \(w_i\) | products.csv | `file_1_view_0` | `Weight` | Capacity consumption per unit for each product |
| \(C\) | capacity.csv | `file_0_view_0` | `Capacity` | Total inventory capacity (765) |

All variables \(x_i\) are non‑negative integers. No additional constraints are imposed beyond the capacity limit.