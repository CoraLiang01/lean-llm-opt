**Mathematical Model**

Let  

* \(I\) be the set of real‑estate areas (indexed by the values of **ProductName**).  
* \(x_i\) – scale of development per day in area \(i\).  

Parameters (obtained from the supplied tables)

* \(\text{Value}_i\) – benefit per unit of development in area \(i\) (column **Value** of *products.csv*).  
* \(\text{Weight}_i\) – resource consumption per unit of development in area \(i\) (column **Weight** of *products.csv*).  
* \(\text{Capacity}\) – overall development capacity limit (column **Capacity** of *capacity.csv*).  

\[
\begin{aligned}
\max\;& \sum_{i\in I} \text{Value}_i \, x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} \text{Weight}_i \, x_i \;\le\; \text{Capacity} \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Capacity}\) | `file_0_view_0` | `Capacity` |
| \(\text{Value}_i\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_i\) | `file_1_view_0` | `Weight` |
| Index set \(I\) | `file_1_view_0` | `ProductName` |

All parameters are taken directly from the corresponding CSV records; no values are enumerated in the model.