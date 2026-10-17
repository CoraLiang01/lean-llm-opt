**Mathematical Model**

Let  

- \(I\) be the set of all products whose *Product Name* contains the string “Baby”.  
- For each \(i\in I\)  

\[
\begin{aligned}
r_i &\;=\;\text{Revenue of product }i,\\
d_i &\;=\;\text{Demand of product }i,\\
I_i &\;=\;\text{Initial Inventory of product }i.
\end{aligned}
\]

Decision variables  

\[
x_i \;\in\; \mathbb{Z}_{+}\quad(\text{number of units of product }i\text{ to sell}),\qquad i\in I.
\]

---

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

---

**Constraints**

1. **Demand limits**  
   \[
   x_i \;\le\; d_i ,\qquad \forall i\in I
   \]

2. **Inventory limits**  
   \[
   x_i \;\le\; I_i ,\qquad \forall i\in I
   \]

3. **Non‑negativity / positivity**  
   \[
   x_i \;\ge\; 1 ,\qquad \forall i\in I
   \]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue of product \(i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand of product \(i\) | `file_0_view_0` | `Demand` |
| \(I_i\) | Initial Inventory of product \(i\) | `file_0_view_0` | `Initial Inventory` |
| Index set \(I\) | Product names containing “Baby” | `file_0_view_0` | `Product Name` |

All parameters are extracted directly from the specified columns of the table `file_0_view_0`; no additional data or assumptions are introduced.