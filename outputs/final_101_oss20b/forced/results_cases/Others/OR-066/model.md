**Mathematical Model**

Let  

* \(I=\{S1,S2\}\) – set of suppliers (rows of *fixed_cost.csv*).  
* \(J=\{C1,C2\}\) – set of supermarkets (rows of *demand.csv*).  

Parameters  

\[
\begin{aligned}
D_j &\;=\;\text{demand of supermarket }j \quad (\text{from file\_0\_view\_0, column } \text{demand})\\
F_i &\;=\;\text{fixed activation cost of supplier }i \quad (\text{from file\_1\_view\_0, column } \text{fixed\_costs})\\
C_{ij}&\;=\;\text{transportation cost per unit from supplier }i\text{ to supermarket }j\\
&\qquad\quad(\text{from file\_2\_view\_0, columns }C1,C2)\\
M &\;=\;\sum_{j\in J} D_j \;=\;360 \quad(\text{big‑M bound})
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\}\quad &&\text{(supplier }i\text{ is activated)}\\
x_{ij}&\ge 0 \quad &&\text{(units shipped from }i\text{ to }j)
\end{aligned}
\]

**Objective**

\[
\min \;\; \sum_{i\in I} F_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} C_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**

\[
\sum_{i\in I} x_{ij}=D_j \qquad \forall j\in J
\]

2. **Activation coupling (big‑M)**  

\[
x_{ij}\;\le\; M\,y_i \qquad \forall i\in I,\; j\in J
\]

3. **Variable domains**

\[
y_i\in\{0,1\}\;\;\forall i\in I,\qquad
x_{ij}\ge 0\;\;\forall i\in I,\; j\in J
\]

---

### Data Mapping

| Parameter | Source Table | Column(s) | Notes |
|-----------|--------------|-----------|-------|
| \(D_j\) | `file_0_view_0` | `demand` | Row identified by `customer` |
| \(F_i\) | `file_1_view_0` | `fixed_costs` | Row identified by `Unnamed: 0` |
| \(C_{ij}\) | `file_2_view_0` | `C1`, `C2` | Row identified by `Unnamed: 0`, column by `customer` |
| \(M\) | Computed | – | Sum of all \(D_j\) |

All indices \(i\) and \(j\) are taken from the full set of rows returned in the observation; no additional filtering is applied. The big‑M value \(M=360\) is sufficient because it equals the total demand and bounds every \(x_{ij}\) when the corresponding supplier is inactive.