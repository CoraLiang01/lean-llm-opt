**Mathematical Model**

Let  

* \(I\) – set of all products whose *Product Name* contains “27in” (obtained from the filtered view).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (USD).  
  * \(d_i\) – deterministic demand (units).  
  * \(s_i\) – initial inventory (units).  

Decision variable  

\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

*\(x_i\)* = number of units of product \(i\) that the retailer chooses to sell.

---

### Objective

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

---

### Constraints

1. **Demand limitation**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limitation**  
   \[
   x_i \;\le\; s_i \qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_i \;\ge\; 0 \qquad \forall i\in I
   \]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(I\) | Product identifiers | `file_0_view_0` | `Product Name` (filtered) |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(s_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

*The filtered view `file_0_view_0` contains only rows where `Product Name` contains the substring “27in”.  Each row supplies the values for the corresponding product \(i\).*