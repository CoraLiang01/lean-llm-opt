**Mathematical Model**

Let  

- \(I=\{W1,\dots ,W10\}\) (warehouse indices)  
- \(J=\{C1,\dots ,C20\}\) (customer indices)

**Parameters**

\[
\begin{aligned}
f_i &\quad\text{fixed opening cost of warehouse }i\\
c_{ij}&\quad\text{unit transportation cost from }i\text{ to }j\\
d_j &\quad\text{demand of customer }j\\
\text{cap}_i &\quad\text{maximum service capacity of warehouse }i
\end{aligned}
\]

**Decision Variables**

\[
\begin{aligned}
x_{ij}&\ge 0 &&\text{quantity shipped from }i\text{ to }j\\
y_i&\in\{0,1\} &&\text{1 if warehouse }i\text{ is opened}
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j,\qquad \forall j\in J
   \]

2. **Capacity limits**  
   \[
   \sum_{j\in J} x_{ij}\le \text{cap}_i\,y_i,\qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_i\in\{0,1\},\qquad \forall i\in I,\; j\in J
   \]

---

### Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(f_i\) | `file_1_view_0` | `Fixed_Cost` |
| \(\text{cap}_i\) | `file_1_view_0` | `Capacity` |
| \(d_j\) | `file_2_view_0` | `Demand` |
| \(c_{ij}\) | `file_0_view_0` | `C1`–`C20` (columns) |

All indices \(i\) and \(j\) are taken from the full set of warehouse and customer identifiers present in the respective tables. No additional numerical values are enumerated; the model references the data via the mapping above.