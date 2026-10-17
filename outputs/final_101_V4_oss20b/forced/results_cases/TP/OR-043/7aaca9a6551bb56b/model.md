**Mathematical Model**

Let  

* \(P\) be the set of drug products (20 items).  
* For each \(p\in P\)  
  * \(v_p\) – benefit per unit (from *products.csv* column **Value**).  
  * \(w_p\) – weight per unit (from *products.csv* column **Weight**).  
* \(C\) – overall stock capacity (from *capacity.csv* column **Capacity**).  

Decision variable  

\[
x_p \;\ge 0 \qquad \forall p\in P
\]

Objective (maximise total benefit)

\[
\max \;\sum_{p\in P} v_p\,x_p
\]

Capacity constraint

\[
\sum_{p\in P} w_p\,x_p \;\le\; C
\]

All other constraints are the non‑negativity of \(x_p\).

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column | Mapping |
|-----------|--------------|---------------|---------|
| \(C\) | `file_0_view_0` | `Capacity` | single scalar value |
| \(v_p\) | `file_1_view_0` | `Value` | one value per product name |
| \(w_p\) | `file_1_view_0` | `Weight` | one value per product name |
| \(P\) | `file_1_view_0` | `ProductName` | set of product identifiers |

The indices \(p\) are taken directly from the **ProductName** column of *products.csv*. No additional aggregation or transformation is applied.