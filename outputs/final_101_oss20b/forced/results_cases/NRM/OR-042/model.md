**Mathematical Model**

Let  

- \(I\) – set of drug types (indexed by the values in column **ProductName** of *products.csv*).  
- \(V_i\) – benefit coefficient for drug type \(i\) (column **Value**).  
- \(W_i\) – weight of one unit of drug type \(i\) (column **Weight**).  
- \(C\) – overall inventory capacity (column **Capacity**).  

Decision variables  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of drug type \(i\) ordered daily.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} V_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} W_i\,x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(V_i\) | `file_1_view_0` | `Value` |
| \(W_i\) | `file_1_view_0` | `Weight` |
| Index set \(I\) | `file_1_view_0` | `ProductName` |

All parameter values are taken directly from the corresponding columns of the specified tables; no additional data or transformations are introduced.