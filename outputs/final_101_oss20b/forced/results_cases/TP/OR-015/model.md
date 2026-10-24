**Mathematical Model**

Let  

* \(I\) – set of Aalop products (obtained from the rows whose *Product Name* starts with “Aalop”).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

(Units of product \(i\) that will be sold.)

---

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

---

**Constraints**

1. **Demand satisfaction**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limitation**  
   \[
   x_i \;\le\; s_i \qquad \forall i\in I
   \]

3. **Non‑negativity** (already stated in the variable definition).

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\)   | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\)   | Demand of product | `file_0_view_0` | `Demand` |
| \(s_i\)   | Initial inventory | `file_0_view_0` | `Initial Inventory` |
| \(I\)     | Product names with prefix “Aalop” | `file_0_view_0` | `Product Name` |

The index set \(I\) is constructed by filtering rows of `file_0_view_0` where the **Product Name** column starts with the string “Aalop”. Each such row defines a unique product index \(i\). The corresponding parameter values are taken directly from the specified columns of that row. No additional data or aggregation is introduced.