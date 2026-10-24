**Mathematical Model**

Let  

- \(I=\{A1,\dots ,A15\}\) (facility sites)  
- \(J=\{B1,\dots ,B8\}\) (distribution centers)

**Parameters**

\[
\begin{aligned}
f_i &\quad\text{fixed cost of opening facility }i,\\
c_{ij}&\quad\text{unit shipping cost from facility }i\text{ to center }j,\\
d_j &\quad\text{demand at center }j,\\
\text{cap}_i &\quad\text{maximum shipment capacity of facility }i.
\end{aligned}
\]

**Decision Variables**

\[
\begin{aligned}
y_i &\in\{0,1\}\quad\text{(facility }i\text{ is opened)} ,\\
x_{ij}&\ge 0\quad\text{(units shipped from }i\text{ to }j).
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j,\qquad \forall j\in J
   \]

2. **Facility capacity**  
   \[
   \sum_{j\in J} x_{ij}\le \text{cap}_i\,y_i,\qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_i\in\{0,1\},\qquad \forall i\in I,\; j\in J
   \]

---

### Data Mapping

| Parameter | Source File | Source Column | Mapping |
|-----------|-------------|---------------|---------|
| \(f_i\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv` | `FixedCost` | `f[i] = FixedCost` |
| \(\text{cap}_i\) | same file | `Capacity` | `cap[i] = Capacity` |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv` | column `B1`–`B8` | `c[i,j] = value in row i, column j` |
| \(d_j\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv` | `Demand` | `d[j] = Demand` |

Indices \(i\) are taken from the `Facility` column of *facility_costs.csv*;  
indices \(j\) are taken from the `Destination` column of *demand_requirements.csv*.  
All parameter values are read directly from the corresponding CSV cells without alteration.