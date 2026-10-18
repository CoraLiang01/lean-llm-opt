##### Model

**Index set**  
\(I=\{1,2,\dots ,100\}\) – product indices.

**Parameters**  
\[
\begin{aligned}
w_i &\;=\;\text{profit per unit for product }i \quad (\text{from } \texttt{profit_per_unit})\\
r1_i &\;=\;\text{resource R1 consumption per unit for product }i \quad (\text{from } \texttt{r1_per_unit})\\
r2_i &\;=\;\text{resource R2 consumption per unit for product }i \quad (\text{from } \texttt{r2_per_unit})\\
r3_i &\;=\;\text{resource R3 consumption per unit for product }i \quad (\text{from } \texttt{r3_per_unit})\\
d_i &\;=\;\text{maximum market demand (units) for product }i \quad (\text{from } \texttt{upper_demand_units})\\
C_{R1} &\;=\;\text{total available amount of resource R1 in the horizon} \quad (\text{from } \texttt{capacity} \text{ where } \texttt{resource}=R1)\\
C_{R2} &\;=\;\text{total available amount of resource R2 in the horizon} \quad (\text{from } \texttt{capacity} \text{ where } \texttt{resource}=R2)\\
C_{R3} &\;=\;\text{total available amount of resource R3 in the horizon} \quad (\text{from } \texttt{capacity} \text{ where } \texttt{resource}=R3)
\end{aligned}
\]

**Decision variables**  
\(x_i \in \mathbb{Z}_{\ge 0}\) – number of 10‑unit batches produced for product \(i\).

**Objective function**  
\[
\max \;\sum_{i\in I} 10\,w_i\,x_i
\]

**Constraints**

1. **Demand limits**  
\[
10\,x_i \;\le\; d_i , \qquad \forall i\in I
\]

2. **Resource constraints**  
\[
\begin{aligned}
\sum_{i\in I} 10\,r1_i\,x_i &\;\le\; C_{R1}\\
\sum_{i\in I} 10\,r2_i\,x_i &\;\le\; C_{R2}\\
\sum_{i\in I} 10\,r3_i\,x_i &\;\le\; C_{R3}
\end{aligned}
\]

3. **Variable domain**  
\[
x_i \in \mathbb{Z}_{\ge 0}, \qquad \forall i\in I
\]

---

##### Data Mapping

| Source | Table ID | Columns |
|--------|----------|---------|
| Product data | `file_0_view_0` | `product`, `profit_per_unit`, `r1_per_unit`, `r2_per_unit`, `r3_per_unit`, `upper_demand_units` |
| Resource capacities | `file_1_view_0` | `resource`, `capacity` |

The parameters \(w_i, r1_i, r2_i, r3_i, d_i\) are instantiated from the rows of `file_0_view_0` for each product \(i\).  
The parameters \(C_{R1}, C_{R2}, C_{R3}\) are taken from the rows of `file_1_view_0` where `resource` equals `R1`, `R2`, and `R3`, respectively.