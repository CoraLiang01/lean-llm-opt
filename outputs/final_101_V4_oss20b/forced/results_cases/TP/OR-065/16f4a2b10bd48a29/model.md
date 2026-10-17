**Mathematical Model**

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij} \;\ge\; d_j \qquad &&\forall j\in J \quad\text{(demand)}\\[4pt]
&x_{ij}\;\le\; M\,y_i \qquad &&\forall i\in I,\; j\in J \quad\text{(linking)}\\[4pt]
&x_{ij}\;\ge\;0 \qquad &&\forall i\in I,\; j\in J\\
&y_i\in\{0,1\} \qquad &&\forall i\in I
\end{aligned}
\]

where  

* \(I=\{S1,S2,S3\}\) – warehouses  
* \(J=\{C1,C2,C3\}\) – customers  
* \(d_j\) – demand of customer \(j\)  
* \(f_i\) – fixed operating cost of warehouse \(i\)  
* \(c_{ij}\) – unit transportation cost from warehouse \(i\) to customer \(j\)  
* \(M=\sum_{j\in J} d_j\) – a sufficiently large constant (upper bound on any shipment)

---

**Data Mapping Rules**

| Parameter | Source Table | Source Columns | Mapping |
|-----------|--------------|----------------|---------|
| \(d_j\) | `file_0_view_0` | `customer`, `demand` | `j = customer`, `d_j = demand` |
| \(f_i\) | `file_1_view_0` | `Unnamed: 0`, `fixed_costs` | `i = Unnamed: 0`, `f_i = fixed_costs` |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0`, `C1`, `C2`, `C3` | `i = Unnamed: 0`, `j = C1/C2/C3`, `c_{ij} = value` |
| \(M\) | Computed | – | \(M = \sum_{j} d_j\) |

All indices, parameters, and variables are defined exactly as above; no additional data or constraints are introduced.