**Mathematical Model**

Let  

* \(I=\{1,\dots ,10\}\) be the set of bread types (index by the order in *products.csv*).  
* \(x_i\in \mathbb{Z}_{\ge 0}\) – number of units of bread type \(i\) ordered each day.  

Parameters (all taken directly from the supplied tables)

* \(v_i\) – expected profit per unit of bread type \(i\) (column **Value** in *products.csv*).  
* \(w_i\) – weight (storage space) per unit of bread type \(i\) (column **Weight** in *products.csv*).  
* \(C\) – total storage capacity (column **Capacity** in *capacity.csv*).  

The bakery’s problem is

\[
\begin{aligned}
\max_{x}\quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Source file | Columns used | Mapping |
|----------|-------------|--------------|---------|
| `file_0_view_0` | `capacity.csv` | `Capacity` | \(C = \text{Capacity}\) |
| `file_1_view_0` | `products.csv` | `ProductName`, `Value`, `Weight` | For each row \(i\): \(v_i = \text{Value}\), \(w_i = \text{Weight}\) |

No additional filtering or aggregation is applied; all rows returned in the observation are used. The decision variables \(x_i\) are indexed by the order of rows in `file_1_view_0`.