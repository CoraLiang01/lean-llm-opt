## Mathematical Model

**Indices**  
\(I\) – set of drug types (the 20 product names in *products.csv*).

**Parameters**  
\(\text{Value}_i\) – benefit coefficient of drug \(i\) (column *Value* in *products.csv*).  
\(\text{Weight}_i\) – weight of one unit of drug \(i\) (column *Weight* in *products.csv*).  
\(C\) – total inventory capacity (column *Capacity* in *capacity.csv*).

**Decision variables**  
\(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of drug \(i\) ordered daily.

**Model**

\[
\begin{aligned}
\max\quad & \sum_{i\in I} \text{Value}_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} \text{Weight}_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0,\; x_i \in \mathbb{Z}\quad \forall i\in I
\end{aligned}
\]

---

## Data Mapping

| Table ID | Source File | Columns Used | Mapping |
|----------|-------------|--------------|---------|
| `file_0_view_0` | `capacity.csv` | `Capacity` | \(C = \text{Capacity}\) |
| `file_1_view_0` | `products.csv` | `ProductName`, `Value`, `Weight` | \(I = \{\text{ProductName}\}\); \(\text{Value}_i = \text{Value}\); \(\text{Weight}_i = \text{Weight}\) |

No additional filtering or aggregation is applied; all rows returned in the observation are used.