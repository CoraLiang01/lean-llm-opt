**Mathematical Model**

Let  

* \(P=\{P1,P2,P3\}\) – set of plants (rows of *plant_capacity.csv*).  
* \(R=\{R1,R2,R3,R4,R5,R6\}\) – set of retailers (columns of *retailer_demand.csv*).  

Parameters  

\[
\begin{aligned}
s_i &\;=\;\text{SupplyCapacity of plant }i \quad (i\in P)\\
d_j &\;=\;\text{Demand of retailer }j \quad (j\in R)\\
c_{ij}&\;=\;\text{Variable cost per carton on route }i\!-\!j \quad (i\in P,j\in R)\\
f_{ij}&\;=\;\text{Fixed activation cost of route }i\!-\!j \quad (i\in P,j\in R)\\
M_{ij}&\;=\;\min\{s_i,d_j\}\quad (i\in P,j\in R)
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
x_{ij}&\ge 0 &&\text{cartons shipped on route }i\!-\!j\\
y_{ij}&\in\{0,1\} &&\text{route }i\!-\!j\text{ is opened}
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in P}\sum_{j\in R}\bigl(c_{ij}\,x_{ij}+f_{ij}\,y_{ij}\bigr)
\]

**Constraints**

1. **Retailer demand satisfaction**  
   \[
   \sum_{i\in P}x_{ij}=d_j \qquad \forall j\in R
   \]

2. **Plant supply limits**  
   \[
   \sum_{j\in R}x_{ij}\le s_i \qquad \forall i\in P
   \]

3. **Route‑activation linking**  
   \[
   x_{ij}\le M_{ij}\,y_{ij}\qquad \forall i\in P,\;j\in R
   \]

4. **Variable‑only non‑negativity**  
   \[
   x_{ij}\ge 0 \qquad \forall i\in P,\;j\in R
   \]

5. **Binary route variables**  
   \[
   y_{ij}\in\{0,1\}\qquad \forall i\in P,\;j\in R
   \]

---

## Data Mapping

| Parameter | Source Table | Source Columns | Mapping Rule |
|-----------|--------------|----------------|--------------|
| \(s_i\) | `file_0_view_0` | `Plant`, `SupplyCapacity` | For each plant row, map `SupplyCapacity` to \(s_i\). |
| \(d_j\) | `file_1_view_0` | `Retailer`, `Demand` | For each retailer row, map `Demand` to \(d_j\). |
| \(c_{ij}\) | `file_2_view_0` | `Plant` (row), `R1`–`R6` (columns) | For each plant row and retailer column, map the cell value to \(c_{ij}\). |
| \(f_{ij}\) | `file_3_view_0` | `Plant` (row), `R1`–`R6` (columns) | For each plant row and retailer column, map the cell value to \(f_{ij}\). |
| \(M_{ij}\) | Computed | – | For each pair \((i,j)\), compute \(M_{ij}=\min\{s_i,d_j\}\). |

*Index sets*  
- \(P\) is the set of all distinct `Plant` values returned from `file_0_view_0`.  
- \(R\) is the set of all distinct `Retailer` values returned from `file_1_view_0`.

All parameters are taken directly from the corresponding source tables; no additional rows or columns are introduced. The model uses the full set of plants and retailers as returned by the data extraction.