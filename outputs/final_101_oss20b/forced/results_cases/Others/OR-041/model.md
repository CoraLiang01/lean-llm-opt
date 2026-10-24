## Mathematical Model

**Indices**  
\(I=\{1,\dots ,20\}\) – set of real‑estate areas (Queens, Brooklyn, …, Ditmars).

**Parameters**  
\[
\begin{aligned}
\text{Value}_i &\;=\; \text{benefit per unit of development in area } i,\\
\text{Weight}_i &\;=\; \text{resource consumption per unit of development in area } i,\\
C &\;=\; \text{overall development capacity}.
\end{aligned}
\]

**Decision Variables**  
\[
x_i \;\ge\; 0 \qquad (i\in I)
\]
scale of development per day in area \(i\).

**Objective**  
\[
\max \;\sum_{i\in I} \text{Value}_i\,x_i
\]

**Capacity Constraint**  
\[
\sum_{i\in I} \text{Weight}_i\,x_i \;\le\; C
\]

---

## Data Mapping

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(C\) | `file_0_view_0` | `Capacity` | Single value 586 |
| \(\text{Value}_i\) | `file_1_view_0` | `Value` | 20 values (e.g., 469 for Queens) |
| \(\text{Weight}_i\) | `file_1_view_0` | `Weight` | 20 values (e.g., 954 for Queens) |

All indices \(i\) correspond to the 20 rows returned from `file_1_view_0`. No additional filtering or aggregation is applied. The model uses the raw values as provided.