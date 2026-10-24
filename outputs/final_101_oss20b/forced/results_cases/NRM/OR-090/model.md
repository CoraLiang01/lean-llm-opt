**Mathematical Model**

Let  

\[
I=\{P_1,\dots ,P_{100}\}
\]

be the set of products.

**Parameters**

\[
\begin{aligned}
&\text{profit}_i      &&\ge 0 &&\text{(profit per unit of product }i)\\
&\text{r1}_i          &&\ge 0 &&\text{(R1 consumption per unit of product }i)\\
&\text{r2}_i          &&\ge 0 &&\text{(R2 consumption per unit of product }i)\\
&\text{r3}_i          &&\ge 0 &&\text{(R3 consumption per unit of product }i)\\
&\text{demand}_i      &&\ge 0 &&\text{(maximum market demand in units for product }i)\\
&\text{batch}_i       &&=10 &&\text{(units per batch for product }i)\\
&\text{cap}_{R1}      &&\ge 0 &&\text{(total available R1)}\\
&\text{cap}_{R2}      &&\ge 0 &&\text{(total available R2)}\\
&\text{cap}_{R3}      &&\ge 0 &&\text{(total available R3)}
\end{aligned}
\]

**Decision Variables**

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

\(x_i\) = number of batches of product \(i\) to produce.

**Objective**

\[
\max \;\; \sum_{i\in I} \text{batch}_i \; \text{profit}_i \; x_i
\]

(Equivalently \(10\,\text{profit}_i\,x_i\) because \(\text{batch}_i=10\).)

**Constraints**

1. **Demand limits**

\[
\text{batch}_i\,x_i \;\le\; \text{demand}_i \qquad \forall i\in I
\]

2. **Resource capacities**

\[
\begin{aligned}
\sum_{i\in I} \text{batch}_i\,\text{r1}_i\,x_i &\le \text{cap}_{R1}\\
\sum_{i\in I} \text{batch}_i\,\text{r2}_i\,x_i &\le \text{cap}_{R2}\\
\sum_{i\in I} \text{batch}_i\,\text{r3}_i\,x_i &\le \text{cap}_{R3}
\end{aligned}
\]

3. **Non‑negativity and integrality**

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

---

## Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| `profit_i` | `file_0_view_0` | `profit_per_unit` |
| `r1_i` | `file_0_view_0` | `r1_per_unit` |
| `r2_i` | `file_0_view_0` | `r2_per_unit` |
| `r3_i` | `file_0_view_0` | `r3_per_unit` |
| `demand_i` | `file_0_view_0` | `upper_demand_units` |
| `batch_i` | `file_0_view_0` | `batch_size_units` |
| `cap_R1` | `file_1_view_0` | `capacity` where `resource = 'R1'` |
| `cap_R2` | `file_1_view_0` | `capacity` where `resource = 'R2'` |
| `cap_R3` | `file_1_view_0` | `capacity` where `resource = 'R3'` |

*All indices \(i\) correspond to the product identifiers in `file_0_view_0` (`P1`–`P100`).*