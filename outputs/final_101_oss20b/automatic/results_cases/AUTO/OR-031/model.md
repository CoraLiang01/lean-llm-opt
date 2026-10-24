**Symbolic Model**

Let  

- \(I\) be the set of dairy products indexed by the values in column **Full_Product_Name** of table **file_0_view_0**.  

Parameters (indexed by \(i\in I\))  

- \(r_i\) – revenue per unit (column **Revenue**)  
- \(d_i\) – demand quantity (column **Demand**)  
- \(inv_i\) – initial inventory (column **Initial Inventory**)  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to fulfill  

**Objective**  

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**  

1. **Demand limit**  
\[
x_i \;\le\; d_i \qquad \forall i\in I
\]

2. **Inventory limit**  
\[
x_i \;\le\; inv_i \qquad \forall i\in I
\]

3. **Non‑negativity & integrality**  
\[
x_i \in \mathbb{Z}_{\ge 0} \qquad \forall i\in I
\]

---

**Data Mapping**

| Symbol | Table ID | Column Name |
|--------|----------|-------------|
| \(I\) | file_0_view_0 | Full_Product_Name |
| \(r_i\) | file_0_view_0 | Revenue |
| \(d_i\) | file_0_view_0 | Demand |
| \(inv_i\) | file_0_view_0 | Initial Inventory |

All parameters are taken directly from the corresponding columns of the supplied CSV view; no values are enumerated or altered.