**Mathematical Model**

Let  

* \(I\) – set of all smartphone models whose *Product Name* contains “TABLET” (the 27 rows returned by the filter).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (column **Revenue**).  
  * \(d_i\) – demand quantity (column **Demand**).  
  * \(v_i\) – initial inventory (column **Initial Inventory**).  

Decision variable  

\[
x_i \in \mathbb{Z}_{+}\quad\text{(units of model }i\text{ to be sold)}
\]

---

### Objective

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

### Constraints

1. **Demand satisfaction**  
   \[
   x_i \le d_i \qquad \forall i\in I
   \]

2. **Inventory limitation**  
   \[
   x_i \le v_i \qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_i \ge 0 \qquad \forall i\in I
   \]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Index set | `file_0_view_0` | `Product Name` (filtered rows) |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(v_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

*The mapping applies to each row returned by the filter; the product name serves as the key for the index set.*