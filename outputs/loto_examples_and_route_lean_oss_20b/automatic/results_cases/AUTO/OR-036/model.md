## Mathematical Model

**Index set**  
\(I = \{\,\text{ProductName}\,\}\)

**Parameters**  
\[
\begin{aligned}
&\text{Value}_i &&\text{(benefit coefficient for product }i\text{)}\\
&\text{Weight}_i &&\text{(inventory weight of product }i\text{)}\\
&\text{Capacity} &&\text{(total inventory capacity)}
\end{aligned}
\]

**Decision variables**  
\[
x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\]

**Objective**  
\[
\max \;\sum_{i\in I} \text{Value}_i\,x_i
\]

**Constraints**  
\[
\sum_{i\in I} \text{Weight}_i\,x_i \;\le\; \text{Capacity}
\]

---

## Data Mapping

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| Capacity | `file_0_view_0` | `Capacity` |
| Value\(_i\) | `file_1_view_0` | `Value` |
| Weight\(_i\) | `file_1_view_0` | `Weight` |

All products listed in `file_1_view_0` constitute the index set \(I\). No additional filtering or transformation is applied.