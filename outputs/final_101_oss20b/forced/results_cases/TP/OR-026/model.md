**Mathematical model**

Let  

* \(I\) – set of all products whose *Product Name* contains the word *Fashion* (obtained from the filtered view).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – deterministic demand (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

*\(x_i\)* = number of units of product \(i\) that will be sold.

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limitation**  
   \[
   x_i \;\le\; s_i \qquad \forall i\in I
   \]

3. **Non‑negativity** (already implicit in variable definition).  

---

**Data‑mapping rules**

| Parameter | Source table | Source column | Mapping rule |
|-----------|--------------|---------------|--------------|
| \(I\) | `file_0_view_0` | `Product Name` | Each distinct value of *Product Name* that contains “Fashion” becomes an index element. |
| \(r_i\) | `file_0_view_0` | `Revenue` | `r_i = float(Revenue)` for the row with product name \(i\). |
| \(d_i\) | `file_0_view_0` | `Demand` | `d_i = int(Demand)` for the row with product name \(i\). |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | `s_i = int(Initial Inventory)` for the row with product name \(i\). |

All numeric values are taken directly from the corresponding CSV cells; no aggregation or transformation beyond the type conversion shown is performed. The model uses only the filtered rows (those whose *Product Name* contains “Fashion”), as specified in the query.