## Mathematical Model

**Index set**  
\(I=\{\text{ProductName}\}\)  (all 25 vehicle types)

**Parameters**  
\[
\begin{aligned}
&\text{Value}_i &&\text{benefit coefficient of product }i \\
&\text{Weight}_i &&\text{inventory weight of product }i \\
&\text{Capacity} &&\text{total inventory capacity}
\end{aligned}
\]

**Decision variables**  
\[
x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\]

**Objective**  
\[
\max \;\sum_{i\in I}\text{Value}_i\,x_i
\]

**Constraint**  
\[
\sum_{i\in I}\text{Weight}_i\,x_i \;\le\; \text{Capacity}
\]

---

## Data Mapping

| Parameter | Source | Column(s) | Notes |
|-----------|--------|-----------|-------|
| `Capacity` | `file_0_view_0` | `Capacity` | single scalar value |
| `Value_i` | `file_1_view_0` | `Value` | mapped by `ProductName` |
| `Weight_i` | `file_1_view_0` | `Weight` | mapped by `ProductName` |
| `I` | `file_1_view_0` | `ProductName` | all 25 rows |

All mappings use the exact column names and row identifiers from the supplied tables. No additional data or constraints are introduced.