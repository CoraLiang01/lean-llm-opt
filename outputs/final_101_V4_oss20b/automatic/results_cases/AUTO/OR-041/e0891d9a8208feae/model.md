**Mathematical Model**

Let  

* \(I\) be the set of real‑estate areas (the 20 `ProductName` values).  
* \(x_i\) – scale of development per day in area \(i\) (non‑negative integer).  

Parameters (all obtained from the supplied tables)

* \(\text{Value}_i\) – benefit per unit of development in area \(i\).  
* \(\text{Weight}_i\) – resource consumption per unit of development in area \(i\).  
* \(\text{Capacity}\) – total available development capacity.

---

\[
\begin{aligned}
\max\;& \sum_{i\in I} \text{Value}_i \, x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} \text{Weight}_i \, x_i \;\le\; \text{Capacity} \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| `Capacity` | `file_0_view_0` | `Capacity` |
| `Value_i` | `file_1_view_0` | `Value` |
| `Weight_i` | `file_1_view_0` | `Weight` |
| Index set `I` | `file_1_view_0` | `ProductName` |

All parameter values are taken directly from the corresponding rows of the tables; no additional data are introduced.