**Mathematical Model**

Let  

* \(I=\{1,\dots ,20\}\) be the set of drug products (one index per row returned from *products.csv*).  
* \(v_i\) – benefit per unit of product \(i\) (from column **Value**).  
* \(w_i\) – weight per unit of product \(i\) (from column **Weight**).  
* \(C\) – total stock‑capacity limit (from column **Capacity** in *capacity.csv*).  

Decision variables  

\[
x_i \in \mathbb{Z}_{+}\qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} v_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

All other constraints are non‑negativity and integrality of \(x_i\).

---

**Data Mapping**

| Table ID | Source file | Columns used | Mapping |
|----------|-------------|--------------|---------|
| `file_0_view_0` | `capacity.csv` | `Capacity` | \(C = \text{Capacity}\) |
| `file_1_view_0` | `products.csv` | `ProductName`, `Value`, `Weight` | For each returned row \(i\): \(v_i = \text{Value}\), \(w_i = \text{Weight}\) |

The index set \(I\) consists of all 20 rows returned from `file_1_view_0`. No additional filtering or aggregation is applied.