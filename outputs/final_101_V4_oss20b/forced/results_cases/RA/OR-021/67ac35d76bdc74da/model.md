**Mathematical Model**

Let  

* \(P\) – set of products (indexed by the values in the *Product Name* column).  
* For each \(p\in P\)  
  * \(R_p\) – revenue per unit (from column **Revenue**).  
  * \(D_p\) – deterministic demand (from column **Demand**).  
  * \(I_p\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_p \in \mathbb{Z}_{\ge 0}\qquad \forall p\in P
\]

---

**Objective**

\[
\max \;\sum_{p\in P} R_p\,x_p
\]

---

**Constraints**

1. **Demand limitation**  
   \[
   x_p \;\le\; D_p \qquad \forall p\in P
   \]

2. **Inventory limitation**  
   \[
   x_p \;\le\; I_p \qquad \forall p\in P
   \]

3. **Non‑negativity & integrality** (already stated in the variable definition).

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(R_p\)   | file_0_view_0 | Revenue |
| \(D_p\)   | file_0_view_0 | Demand |
| \(I_p\)   | file_0_view_0 | Initial Inventory |
| Index \(p\) | file_0_view_0 | Product Name |

All parameters are taken directly from the runtime records of **Salesofsummerclothes.csv** (table id `file_0_view_0`). No additional data or constraints are introduced beyond those specified in the query.