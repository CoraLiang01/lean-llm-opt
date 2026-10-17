**Mathematical Model**

Let  

* \(I\) – set of 4U products (identified by *Product Name*).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad\forall i\in I
\]

---

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

---

**Constraints**

1. **Demand limit**  
   \[
   x_i \;\le\; d_i \qquad\forall i\in I
   \]

2. **Inventory limit**  
   \[
   x_i \;\le\; s_i \qquad\forall i\in I
   \]

3. **Non‑negativity & integrality**  
   \[
   x_i \in \mathbb{Z}_{\ge 0}\qquad\forall i\in I
   \]

---

**Data Mapping Rules**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(I\) (index set) | Product identifiers | `file_0_view_0` | `Product Name` |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(s_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

*All values are taken directly from the filtered rows where **Product Name** contains the substring `"4U"`.*