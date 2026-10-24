**Symbolic Model**

Let  

- \(F=\{A1,\dots ,A15\}\) : set of potential factory sites  
- \(D=\{B1,\dots ,B8\}\) : set of distribution centers  

Parameters  

- \(f_i\) : fixed construction cost of factory \(i\)  (\(i\in F\))  
- \(cap_i\) : production capacity of factory \(i\)  (\(i\in F\))  
- \(c_{ij}\) : unit shipping cost from factory \(i\) to center \(j\) (\(i\in F,\;j\in D\))  
- \(d_j\) : demand at center \(j\)  (\(j\in D\))

Decision variables  

- \(y_i\in\{0,1\}\) : 1 if factory \(i\) is built, 0 otherwise  
- \(x_{ij}\in\mathbb{Z}_{+}\) : units shipped from \(i\) to \(j\)

---

\[
\begin{aligned}
\min\;& \sum_{i\in F} f_i\,y_i \;+\; \sum_{i\in F}\sum_{j\in D} c_{ij}\,x_{ij} \\
\text{s.t. } & \sum_{j\in D} x_{ij} \;\le\; cap_i\,y_i \qquad &&\forall i\in F \\
& \sum_{i\in F} x_{ij} \;=\; d_j \qquad &&\forall j\in D \\
& x_{ij}\;\ge\;0,\; x_{ij}\in\mathbb{Z} \qquad &&\forall i\in F,\;j\in D \\
& y_i\in\{0,1\} \qquad &&\forall i\in F
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Table ID | Column(s) |
|-----------|--------------|----------|-----------|
| \(f_i\) | `facility_costs.csv` | `file_0_view_0` | `Facility`, `FixedCost` |
| \(cap_i\) | `facility_costs.csv` | `file_0_view_0` | `Facility`, `Capacity` |
| \(c_{ij}\) | `shipping_costs.csv` | `file_1_view_0` | `Origin` (facility), `B1`–`B8` (destinations) |
| \(d_j\) | `demand_requirements.csv` | `file_2_view_0` | `Destination`, `Demand` |

All parameter values are taken directly from the corresponding columns in the listed tables; no additional data or assumptions are introduced.