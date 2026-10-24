**Mathematical Model**

Let  

- \(F=\{A1,\dots ,A15\}\) : set of potential factory sites  
- \(D=\{B1,\dots ,B8\}\) : set of distribution centers  

Parameters  

\[
\begin{aligned}
f_i &\quad\text{fixed cost of building facility }i\in F\\
c_{ij}&\quad\text{variable shipping cost per unit from }i\in F\text{ to }j\in D\\
\text{cap}_i &\quad\text{maximum production capacity of facility }i\in F\\
d_j &\quad\text{demand at distribution center }j\in D
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\}\quad\text{(1 if facility }i\text{ is built)}\\
x_{ij} &\ge 0\quad\text{(units shipped from }i\text{ to }j)
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in F} f_i\,y_i \;+\; \sum_{i\in F}\sum_{j\in D} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Capacity limits**

\[
\sum_{j\in D} x_{ij}\;\le\; \text{cap}_i\,y_i \qquad \forall i\in F
\]

2. **Demand satisfaction**

\[
\sum_{i\in F} x_{ij}\;=\; d_j \qquad \forall j\in D
\]

3. **Variable domains**

\[
x_{ij}\ge 0 \quad\forall i\in F,\;j\in D
\]

\[
y_i\in\{0,1\}\quad\forall i\in F
\]

---

**Data Mapping**

| Parameter | Source Table | Table ID | Column(s) |
|-----------|--------------|----------|-----------|
| \(f_i\) | `facility_costs.csv` | `file_0_view_0` | `Facility`, `FixedCost` |
| \(\text{cap}_i\) | `facility_costs.csv` | `file_0_view_0` | `Facility`, `Capacity` |
| \(c_{ij}\) | `shipping_costs.csv` | `file_1_view_0` | `Origin`, `B1`–`B8` |
| \(d_j\) | `demand_requirements.csv` | `file_2_view_0` | `Destination`, `Demand` |

*Indices \(i\) and \(j\) are mapped to the corresponding values in the `Facility` and `Destination` columns, respectively.*