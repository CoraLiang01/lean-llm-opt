**Mathematical Model**

Sets  
\[
I=\{S_1,S_2,S_3,S_4,S_5\},\qquad  
J=\{C_1,C_2,C_3,C_4,C_5\}
\]

Parameters  
\[
\begin{aligned}
f_i &\;=\;\text{fixed cost of opening supplier }i,\\
c_{ij}&\;=\;\text{transportation cost per unit from supplier }i\text{ to branch }j,\\
d_j &\;=\;\text{demand of branch }j .
\end{aligned}
\]

Decision variables  
\[
\begin{aligned}
y_i &\in\{0,1\}\quad &&\text{(supplier }i\text{ is open)}\\
x_{ij}&\ge 0 \quad &&\text{(units supplied from }i\text{ to }j)
\end{aligned}
\]

Objective  
\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
\[
\sum_{i\in I} x_{ij}=d_j \qquad \forall j\in J
\]

2. **Supplier activation**  
\[
x_{ij}\;\le\; d_j\,y_i \qquad \forall i\in I,\; j\in J
\]

3. **Non‑negativity**  
\[
x_{ij}\ge 0 \qquad \forall i\in I,\; j\in J
\]

---

### Data Mapping

| Parameter | Source Table | Row Identifier | Column Identifier | Notes |
|-----------|--------------|----------------|-------------------|-------|
| \(f_i\) | `file_1_view_0` (`fixed_cost.csv`) | `Unnamed: 0` (supplier id) | `fixed_costs` | Fixed cost per supplier |
| \(c_{ij}\) | `file_2_view_0` (`transportation_costs.csv`) | `Unnamed: 0` (supplier id) | `C1`–`C5` (branch ids) | Transportation cost per unit |
| \(d_j\) | `file_0_view_0` (`demand.csv`) | `customer` (branch id) | `demand` | Demand per branch |

The index sets \(I\) and \(J\) are taken directly from the row identifiers of the fixed‑cost and transportation‑cost matrices, and from the column identifiers of the transportation‑cost matrix, respectively. No additional filtering or aggregation is applied.