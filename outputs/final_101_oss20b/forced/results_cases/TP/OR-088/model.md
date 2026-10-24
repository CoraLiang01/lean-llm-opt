**Mathematical model**

Let  

* \(I=\{F1,\dots ,F15\}\) – set of candidate plants  
* \(J=\{C1,\dots ,C15\}\) – set of customers  

Parameters (all real, non‑negative)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(f_i\) | fixed opening cost of plant \(i\) | `file_0_view_0` column *fixed_cost* |
| \(s_i\) | capacity of plant \(i\) | `file_0_view_0` column *capacity* |
| \(c_{ij}\) | per‑unit transport cost from plant \(i\) to customer \(j\) | `file_0_view_0` columns *C1*…*C15* |
| \(d_j\) | demand of customer \(j\) | `file_1_view_0` column *demand* |

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if plant \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity shipped from plant \(i\) to customer \(j\)

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij} \;\ge\; d_j , && \forall j\in J \quad\text{(demand)}\\[4pt]
& \sum_{j\in J} x_{ij} \;\le\; s_i\,y_i , && \forall i\in I \quad\text{(capacity)}\\[4pt]
& y_i \in \{0,1\}, && \forall i\in I\\
& x_{ij}\ge 0 , && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data‑mapping rules**

| Table ID | Column | Parameter | Mapping rule |
|----------|--------|-----------|--------------|
| `file_0_view_0` | `plant` | plant index \(i\) | `i = plant` |
| `file_0_view_0` | `fixed_cost` | \(f_i\) | `f_i = fixed_cost` |
| `file_0_view_0` | `capacity` | \(s_i\) | `s_i = capacity` |
| `file_0_view_0` | `Ck` (k=1..15) | \(c_{iCk}\) | `c_{i,Ck} = Ck` |
| `file_1_view_0` | `customer` | customer index \(j\) | `j = customer` |
| `file_1_view_0` | `demand` | \(d_j\) | `d_j = demand` |

These rules associate each row of the CSV views with the corresponding model parameters, preserving the original identifiers and values. No additional data are introduced or aggregated.