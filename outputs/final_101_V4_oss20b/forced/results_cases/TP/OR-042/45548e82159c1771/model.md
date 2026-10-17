**Mathematical Model**

Let  

- \(I\) be the set of drug types indexed by the values of **ProductName** in *products.csv*.  
- \(x_i \in \mathbb{Z}_{\ge 0}\) be the daily order quantity of drug type \(i\).  

Parameters (all obtained from the supplied tables):

- \(v_i\) – benefit coefficient of drug \(i\) (column **Value** in *products.csv*).  
- \(w_i\) – weight per unit of drug \(i\) (column **Weight** in *products.csv*).  
- \(C\) – overall inventory capacity (column **Capacity** in *capacity.csv*).  

The optimization problem is

\[
\begin{aligned}
\max_{x}\quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C, \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}\quad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(C\) | `file_0_view_0` | `Capacity` | `C = Capacity` |
| \(v_i\) | `file_1_view_0` | `Value` | `v[ProductName] = Value` |
| \(w_i\) | `file_1_view_0` | `Weight` | `w[ProductName] = Weight` |
| Index set \(I\) | `file_1_view_0` | `ProductName` | `I = {ProductName}` |

All numeric values are taken directly from the corresponding CSV records; no aggregation or transformation is performed beyond the mappings shown.