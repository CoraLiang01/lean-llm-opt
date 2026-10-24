**Mathematical Model**

Let  

- \(I=\{P1,\dots ,P100\}\) (product indices)  
- \(R=\{R1,R2,R3\}\) (resource indices)  

Decision variable  

\[
x_i \in \mathbb{Z}_{+}\qquad \forall i\in I
\]

( number of 10‑unit batches produced of product \(i\) ).

Parameters (obtained from the CSV files)

\[
\begin{aligned}
&\text{profit}_{i} &&\text{(profit per unit of product }i)\\
&r_{1,i},\, r_{2,i},\, r_{3,i} &&\text{(resource consumption per unit of product }i)\\
&\text{demand}_{i} &&\text{(maximum market demand in units for product }i)\\
&\text{batch}_i &&\text{(batch size in units for product }i)\\
&\text{cap}_{k} &&\text{(total available amount of resource }k)
\end{aligned}
\]

All parameters are non‑negative real numbers; \(\text{batch}_i=10\) for every \(i\).

---

\[
\begin{aligned}
\max\;&\sum_{i\in I} 10\,\text{profit}_{i}\,x_i \\[4pt]
\text{s.t. } &10\,x_i \;\le\; \text{demand}_{i}\qquad \forall i\in I \\[4pt]
&\sum_{i\in I} 10\,r_{k,i}\,x_i \;\le\; \text{cap}_{k}\qquad \forall k\in R \\[4pt]
&x_i \in \mathbb{Z}_{+}\qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Table ID | Column | Parameter | Index |
|----------|--------|-----------|-------|
| `file_0_view_0` | `product` | `i` | product index |
| `file_0_view_0` | `profit_per_unit` | `profit_i` | product |
| `file_0_view_0` | `r1_per_unit` | `r1_i` | product |
| `file_0_view_0` | `r2_per_unit` | `r2_i` | product |
| `file_0_view_0` | `r3_per_unit` | `r3_i` | product |
| `file_0_view_0` | `upper_demand_units` | `demand_i` | product |
| `file_0_view_0` | `batch_size_units` | `batch_i` | product |
| `file_1_view_0` | `resource` | `k` | resource index |
| `file_1_view_0` | `capacity` | `cap_k` | resource |

All indices and parameters are taken directly from the complete runtime records of the two CSV files. No additional data or aggregation is introduced.