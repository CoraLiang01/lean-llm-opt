**Mathematical Model**

Let  

- \(I=\{A1,\dots ,A15\}\) (Factory sites)  
- \(J=\{B1,\dots ,B8\}\) (Distribution centers)

Parameters  

\[
\begin{aligned}
f_i &\quad\text{fixed cost of building factory }i, && i\in I\\
c_{ij}&\quad\text{variable shipping cost per unit from }i\text{ to }j, && i\in I,\;j\in J\\
d_j &\quad\text{demand at destination }j, && j\in J\\
\text{cap}_i &\quad\text{maximum production capacity of factory }i, && i\in I
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\}\quad &&\text{factory }i\text{ is built} \\
x_{ij} &\ge 0 \quad &&\text{units shipped from }i\text{ to }j
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j ,\qquad \forall j\in J
   \]

2. **Capacity (only if factory built)**  
   \[
   \sum_{j\in J} x_{ij}\;\le\;\text{cap}_i\,y_i ,\qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_i\in\{0,1\}
   \]

---

### Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(f_i\) | `file_0_view_0` | `FixedCost` |
| \(\text{cap}_i\) | `file_0_view_0` | `Capacity` |
| \(c_{ij}\) | `file_1_view_0` | `B1`–`B8` (columns indexed by destination) |
| \(d_j\) | `file_2_view_0` | `Demand` |

All index sets \(I\) and \(J\) are taken from the row identifiers of the respective tables (`Facility` column in `file_0_view_0` and `Destination` column in `file_2_view_0`). No additional constraints or parameters are introduced beyond those explicitly required by the problem statement.