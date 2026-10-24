**Mathematical Model**

Let  

- \(I\) be the set of all products whose SKU starts with “ZZ”.  
- For each \(i\in I\) let  

  \[
  \text{Revenue}_i \;(\$),\qquad 
  \text{Demand}_i \;( \text{units}),\qquad 
  \text{InitialInventory}_i \;( \text{units})
  \]

  be the corresponding parameters.

- Decision variable  

  \[
  x_i \in \mathbb{Z}_{\ge 0}\quad\text{(units of product }i\text{ to sell)}.
  \]

---

### Objective
\[
\max \;\sum_{i\in I} \text{Revenue}_i \; x_i
\]

### Constraints
\[
\begin{aligned}
&x_i \;\le\; \text{Demand}_i &&\forall i\in I \quad &&(\text{cannot sell more than demand})\\[4pt]
&x_i \;\le\; \text{InitialInventory}_i &&\forall i\in I \quad &&(\text{cannot sell more than stock})\\[4pt]
&x_i \;\ge\; 0 &&\forall i\in I \quad &&(\text{non‑negative integer})
\end{aligned}
\]

---

## Data Mapping

| Table ID | Column | Maps to |
|----------|--------|---------|
| `file_0_view_0` | `SKU` | Index set \(I\) (each SKU value is an element of \(I\)) |
| `file_0_view_0` | `Revenue` | Parameter \(\text{Revenue}_i\) |
| `file_0_view_0` | `Demand` | Parameter \(\text{Demand}_i\) |
| `file_0_view_0` | `Initial Inventory` | Parameter \(\text{InitialInventory}_i\) |

Each row of `file_0_view_0` supplies the values for the corresponding product \(i\). No other columns are used.