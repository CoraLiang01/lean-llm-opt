**Mathematical Model**

Let  

* \(I\) be the set of all product types (the 10 rows in *products.csv*).  
* For each \(i\in I\)  
  * \(w_i\) – weight of one unit of product \(i\) (kg)  
  * \(v_i\) – benefit (value) of one unit of product \(i\)  
* \(C\) – total inventory capacity (kg) from *capacity.csv*  

Decision variables  

* \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) ordered daily  

Objective  

\[
\max \;\sum_{i\in I} v_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

Non‑negativity and integrality  

\[
x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(C\) | Total capacity | `file_0_view_0` | `Capacity` |
| \(w_i\) | Weight of product \(i\) | `file_1_view_0` | `Weight` |
| \(v_i\) | Value of product \(i\) | `file_1_view_0` | `Value` |
| \(I\) | Product names | `file_1_view_0` | `ProductName` |

All parameters are taken directly from the corresponding columns of the specified tables; no additional filtering or transformation is applied. The index set \(I\) consists of every row in *products.csv*.