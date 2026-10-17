**Mathematical Model**

Let  

* \(F=\{A1,\dots ,A15\}\) – set of potential factory sites  
* \(D=\{B1,\dots ,B8\}\) – set of distribution centers  

Parameters  

\[
\begin{aligned}
f_i      &\quad\text{fixed cost of building facility }i, && i\in F\\
c_{ij}   &\quad\text{variable shipping cost per unit from }i\text{ to }j, && i\in F,\; j\in D\\
\text{Cap}_i &\quad\text{maximum production capacity of facility }i, && i\in F\\
d_j      &\quad\text{demand at destination }j, && j\in D
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in \{0,1\} &&\text{(1 if facility }i\text{ is built)}\\
x_{ij} &\in \mathbb{Z}_{\ge 0} &&\text{units shipped from }i\text{ to }j
\end{aligned}
\]

---

### Objective  

\[
\min \;\; \sum_{i\in F} f_i\,y_i \;+\; \sum_{i\in F}\sum_{j\in D} c_{ij}\,x_{ij}
\]

### Constraints  

1. **Facility capacity**  
\[
\sum_{j\in D} x_{ij}\;\le\; \text{Cap}_i\,y_i \qquad \forall i\in F
\]

2. **Demand satisfaction**  
\[
\sum_{i\in F} x_{ij}\;\ge\; d_j \qquad \forall j\in D
\]

3. **Non‑negativity & integrality**  
\[
x_{ij}\;\ge\;0,\; x_{ij}\in\mathbb{Z}\qquad \forall i\in F,\; j\in D
\]
\[
y_i\in\{0,1\}\qquad \forall i\in F
\]

---

## Data Mapping

| Parameter | Table ID | Column(s) | Index ID column |
|-----------|----------|-----------|-----------------|
| \(f_i\) | `file_0_view_0` | `FixedCost` | `Facility` |
| \(\text{Cap}_i\) | `file_0_view_0` | `Capacity` | `Facility` |
| \(c_{ij}\) | `file_1_view_0` | `B1`–`B8` | `Origin` (facility) |
| \(d_j\) | `file_2_view_0` | `Demand` | `Destination` |

*Index sets*  
- \(F\) derived from the distinct values in column `Facility` of `file_0_view_0`.  
- \(D\) derived from the distinct values in column `Destination` of `file_2_view_0`.  

All parameters are taken directly from the supplied CSV tables; no values are enumerated or altered.