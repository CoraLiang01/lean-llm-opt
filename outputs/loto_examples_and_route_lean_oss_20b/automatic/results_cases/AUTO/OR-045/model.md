**Mathematical Model**

Let  

* \(I\) be the set of all product types (the 10 rows of *products.csv*).  
* For each \(i\in I\):  
  * \(w_i\) – weight of one unit of product \(i\) (kg) – taken from column **Weight** of *products.csv*.  
  * \(v_i\) – benefit (value) of one unit of product \(i\) – taken from column **Value** of *products.csv*.  
* \(C\) – total inventory capacity (kg) – taken from column **Capacity** of *capacity.csv*.  

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} v_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

All other constraints are the non‑negativity and integrality of \(x_i\).

---

**Data Mapping**

| Table ID | Source File | Columns Used | Mapping |
|----------|-------------|--------------|---------|
| `file_0_view_0` | `capacity.csv` | `Capacity` | \(C = \text{Capacity}\) |
| `file_1_view_0` | `products.csv` | `ProductName`, `Weight`, `Value` | For each row \(i\): \(w_i = \text{Weight}\), \(v_i = \text{Value}\) |

The index set \(I\) is defined as the set of all `ProductName` values returned in `file_1_view_0`. No additional filtering is applied.