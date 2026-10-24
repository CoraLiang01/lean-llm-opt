**Mathematical Model**

Let  

* \(P\) be the set of all products (identified by *Product Name*).  
* For each \(p\in P\)  
  * \(rev_p\) – revenue per unit (from column **Revenue**).  
  * \(dem_p\) – demand quantity (from column **Demand**).  
  * \(inv_p\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_p \;\ge\; 0 \qquad \forall\,p\in P
\]

*\(x_p\)* is the quantity of product \(p\) that the bakery plans to sell.

---

**Objective**

\[
\max \;\sum_{p\in P} rev_p \, x_p
\]

---

**Constraints**

1. **Demand satisfaction**  
   \[
   x_p \;\le\; dem_p \qquad \forall\,p\in P
   \]

2. **Inventory limitation**  
   \[
   x_p \;\le\; inv_p \qquad \forall\,p\in P
   \]

---

**Data Mapping**

| Parameter | Source column | Table ID | Index column |
|-----------|---------------|----------|--------------|
| \(rev_p\) | Revenue | `file_0_view_0` | Product Name |
| \(dem_p\) | Demand | `file_0_view_0` | Product Name |
| \(inv_p\) | Initial Inventory | `file_0_view_0` | Product Name |

All parameters are taken directly from the CSV file `Frenchbakerydailysales.csv`; no numerical values are enumerated in the model.