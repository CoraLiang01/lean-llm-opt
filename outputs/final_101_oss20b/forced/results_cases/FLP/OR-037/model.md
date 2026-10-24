**Mathematical Model**

Let  

- \(I\) be the set of product types (indexed by *ProductName*).  
- \(p_i\) be the profit per unit of product \(i\) (from column *Value*).  
- \(w_i\) be the weight (inventory space) per unit of product \(i\) (from column *Weight*).  
- \(C\) be the total inventory capacity (from column *Capacity*).  
- \(x_i\) be the number of units of product \(i\) ordered per day.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(C\) | Inventory capacity | `file_0_view_0` | `Capacity` |
| \(I\) | Product identifiers | `file_1_view_0` | `ProductName` |
| \(p_i\) | Profit per unit | `file_1_view_0` | `Value` |
| \(w_i\) | Weight (space) per unit | `file_1_view_0` | `Weight` |

All parameters are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.