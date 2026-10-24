**Mathematical Model**

Let  

- \(I=\{A1,\dots ,A15\}\) (Factory sites)  
- \(J=\{B1,\dots ,B8\}\) (Distribution centers)

**Decision Variables**  

\[
\begin{aligned}
y_i &\in \{0,1\} &&\text{(factory \(i\) built)}\\
x_{ij} &\ge 0 &&\text{(units shipped from \(i\) to \(j\))}
\end{aligned}
\]

**Parameters**  

\[
\begin{aligned}
f_i &\ge 0 &&\text{fixed cost of factory }i\\
c_{ij} &\ge 0 &&\text{variable shipping cost per unit from }i\text{ to }j\\
d_j &\ge 0 &&\text{demand at destination }j\\
\text{cap}_i &\ge 0 &&\text{maximum production capacity of factory }i
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j \quad \forall j\in J
   \]

2. **Capacity (only if factory built)**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; \text{cap}_i\,y_i \quad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_i\in\{0,1\}\quad \forall i\in I,\; j\in J
   \]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(f_i\) | `file_0_view_0` | `FixedCost` |
| \(\text{cap}_i\) | `file_0_view_0` | `Capacity` |
| \(c_{ij}\) | `file_1_view_0` | `B1`–`B8` (columns) |
| \(d_j\) | `file_2_view_0` | `Demand` |

All indices \(i\) and \(j\) are taken from the row and column identifiers of the respective tables (`Facility` and `Destination`). No numerical values are enumerated; the model references the data through the mapping above.