**Mathematical Model**

Let  

* \(I=\{D1,D2,D3,D4\}\) (Depots)  
* \(J=\{M1,M2,M3,M4,M5\}\) (Markets)

Parameters  

\[
\begin{aligned}
s_i &\quad\text{(Depot supply capacity)}\\
d_j &\quad\text{(Market demand)}\\
c_{ij} &\quad\text{(Per‑unit variable shipping cost)}\\
f_{ij} &\quad\text{(Fixed activation cost)}\\
M_{ij} &\quad\text{(Big‑M, }M_{ij}=\min(s_i,d_j)\text{)}
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
x_{ij}&\ge 0 &&\text{(shipment quantity)}\\
y_{ij}&\in\{0,1\} &&\text{(route activation)}
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I}\sum_{j\in J}\bigl(c_{ij}\,x_{ij}+f_{ij}\,y_{ij}\bigr)
\]

**Constraints**

1. **Market demand satisfaction**

\[
\sum_{i\in I}x_{ij}=d_j \qquad \forall j\in J
\]

2. **Depot supply limits**

\[
\sum_{j\in J}x_{ij}\le s_i \qquad \forall i\in I
\]

3. **Linking shipment to activation**

\[
x_{ij}\le M_{ij}\,y_{ij}\qquad \forall i\in I,\;j\in J
\]

4. **Variable bounds**

\[
x_{ij}\ge 0,\qquad y_{ij}\in\{0,1\}\qquad \forall i\in I,\;j\in J
\]

---

## Data Mapping

| Parameter | Source Table | Source Columns | Notes |
|-----------|--------------|----------------|-------|
| \(s_i\) | `file_0_view_0` | `Depot`, `SupplyCapacity` | One row per depot |
| \(d_j\) | `file_1_view_0` | `Market`, `Demand` | One row per market |
| \(c_{ij}\) | `file_2_view_0` | `Depot`, `M1`–`M5` | Matrix of per‑unit costs |
| \(f_{ij}\) | `file_3_view_0` | `Depot`, `M1`–`M5` | Matrix of fixed activation costs |
| \(M_{ij}\) | Derived | – | \(M_{ij}=\min(s_i,d_j)\) |

All index sets \(I\) and \(J\) are taken from the full set of rows in the corresponding source tables; no additional filtering is applied. The model uses the exact values as supplied in the CSV files. No numerical values are enumerated in the formulation.