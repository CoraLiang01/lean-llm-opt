## Mathematical Model

**Indices**  
\(I\) – set of drug types (the 20 product names in *products.csv*).

**Parameters**  
\[
\begin{aligned}
&\text{Value}_i &&\text{(benefit coefficient)} \quad (i\in I)\\
&\text{Weight}_i &&\text{(unit weight)} \quad (i\in I)\\
&\text{Capacity} &&\text{(overall inventory capacity)} 
\end{aligned}
\]

**Decision variables**  
\[
x_i \in \mathbb{Z}_{\ge 0}\quad (i\in I)
\]

**Objective**  
\[
\max \;\sum_{i\in I} \text{Value}_i\,x_i
\]

**Constraint**  
\[
\sum_{i\in I} \text{Weight}_i\,x_i \;\le\; \text{Capacity}
\]

---

## Data Mapping

| Parameter | Source Table | Columns | Mapping |
|-----------|--------------|---------|---------|
| `Capacity` | `file_0_view_0` | `Capacity` | single scalar value |
| `Value_i` | `file_1_view_0` | `ProductName`, `Value` | `Value_i = Value` for each row |
| `Weight_i` | `file_1_view_0` | `ProductName`, `Weight` | `Weight_i = Weight` for each row |

*All rows in `file_1_view_0` are included; no filtering is applied.*