## Mathematical Model

Let  

* \(I=\{P_1,\dots ,P_{100}\}\) – set of products  
* \(R=\{R1,R2,R3\}\) – set of resources  

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

( number of 10‑unit batches produced of product \(i\) ).

Parameters  

\[
\begin{aligned}
&\text{profit}_{i}      &&\text{(profit per unit, from file_0_view_0)}\\
&\text{r1}_{i}          &&\text{(R1 consumption per unit, from file_0_view_0)}\\
&\text{r2}_{i}          &&\text{(R2 consumption per unit, from file_0_view_0)}\\
&\text{r3}_{i}          &&\text{(R3 consumption per unit, from file_0_view_0)}\\
&\text{demand}_{i}      &&\text{(upper demand in units, from file_0_view_0)}\\
&\text{cap}_{r}         &&\text{(resource capacity, from file_1_view_0)}\\
&\text{batch} = 10      &&\text{(fixed batch size, from file_0_view_0)}
\end{aligned}
\]

### Objective

\[
\max \;\sum_{i\in I} \text{batch}\;x_i\;\text{profit}_{i}
\]

### Constraints

1. **Resource limits**

\[
\sum_{i\in I} \text{batch}\;x_i\;\text{r1}_{i}\;\le\;\text{cap}_{R1}\qquad
\sum_{i\in I} \text{batch}\;x_i\;\text{r2}_{i}\;\le\;\text{cap}_{R2}\qquad
\sum_{i\in I} \text{batch}\;x_i\;\text{r3}_{i}\;\le\;\text{cap}_{R3}
\]

2. **Demand limits**

\[
\text{batch}\;x_i \;\le\; \text{demand}_{i}\qquad \forall i\in I
\]

3. **Non‑negativity & integrality**

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

---

## Data Mapping

| Parameter | Source Table | Column | Notes |
|-----------|--------------|--------|-------|
| `profit_i` | `file_0_view_0` | `profit_per_unit` | per unit profit |
| `r1_i` | `file_0_view_0` | `r1_per_unit` | R1 consumption per unit |
| `r2_i` | `file_0_view_0` | `r2_per_unit` | R2 consumption per unit |
| `r3_i` | `file_0_view_0` | `r3_per_unit` | R3 consumption per unit |
| `demand_i` | `file_0_view_0` | `upper_demand_units` | maximum market demand (units) |
| `batch` | `file_0_view_0` | `batch_size_units` | fixed at 10 |
| `cap_R1` | `file_1_view_0` | `capacity` where `resource = 'R1'` | total R1 capacity |
| `cap_R2` | `file_1_view_0` | `capacity` where `resource = 'R2'` | total R2 capacity |
| `cap_R3` | `file_1_view_0` | `capacity` where `resource = 'R3'` | total R3 capacity |

All indices \(i\) are taken from the 100 product rows in `file_0_view_0`; all resource indices \(r\) are taken from the 3 rows in `file_1_view_0`. No additional rows or columns are introduced.