**Mathematical Model**

Let  

- \(I\) – set of all SKUs whose code starts with the prefix **ZZ** (obtained from the filtered view).  
- For each \(i\in I\)  
  - \(r_i\) – revenue per unit (from column **Revenue**).  
  - \(d_i\) – total demand over the horizon (from column **Demand**).  
  - \(v_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

- \(x_i\) – number of units of SKU \(i\) the store intends to sell.  
  \[
  x_i \in \mathbb{Z}_{+}\quad(\text{positive integer})
  \]

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

1. **Demand limit**  
   \[
   x_i \le d_i \qquad \forall i\in I
   \]

2. **Inventory limit**  
   \[
   x_i \le v_i \qquad \forall i\in I
   \]

3. **Non‑negativity (implicit in domain)**  
   \[
   x_i \ge 1 \qquad \forall i\in I
   \]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Total demand | `file_0_view_0` | `Demand` |
| \(v_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |
| \(I\) | Set of SKUs with prefix “ZZ” | `file_0_view_0` | `SKU` (filter: prefix “ZZ”) |

All parameter values are taken directly from the filtered rows of the view `file_0_view_0`; no additional data or assumptions are introduced.