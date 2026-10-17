**Mathematical Model**

Let  

- \(I\) be the set of drug products (indexed by the *ProductName* values in *products.csv*).  
- For each \(i\in I\)  
  - \(v_i\)  – benefit (Value) of product \(i\) (units of benefit per unit ordered).  
  - \(w_i\)  – weight (Space/Volume) of product \(i\) (units of capacity consumed per unit ordered).  
- \(C\) – total available stock capacity (Capacity) from *capacity.csv*.  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to order each day.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(C\)   | `file_0_view_0` | `Capacity` |

The index set \(I\) is derived from the distinct `ProductName` values in `file_1_view_0`. No other data or constraints are introduced beyond those specified in the query.