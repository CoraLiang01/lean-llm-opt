**Mathematical Model**

Let  

- \(I=\{i\}\) be the set of suppliers (rows of *fixed_cost.csv*).  
- \(J=\{j\}\) be the set of supermarkets (columns of *demand.csv*).  

Parameters  

\[
\begin{aligned}
f_i &\;=\;\text{fixed cost of supplier }i \quad (i\in I)\\
c_{ij}&\;=\;\text{transportation cost per unit from supplier }i\text{ to supermarket }j \quad (i\in I,\;j\in J)\\
d_j &\;=\;\text{demand of supermarket }j \quad (j\in J)\\
M &\;=\;\sum_{j\in J} d_j \quad\text{(big‑M upper bound)}
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\}\quad\text{(supplier }i\text{ is open)}\\
x_{ij} &\ge 0 \quad\text{(units shipped from }i\text{ to }j)
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

2. **Supplier activation (big‑M)**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; M\,y_i \qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   y_i\in\{0,1\},\; x_{ij}\ge 0
   \]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(I\) | `file_1_view_0` | `Unnamed: 0` | Supplier identifiers |
| \(J\) | `file_0_view_0` | `customer` | Supermarket identifiers |
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Fixed cost per supplier |
| \(d_j\) | `file_0_view_0` | `demand` | Demand per supermarket |
| \(c_{ij}\) | `file_2_view_0` | matrix of `C1`–`C25` | Transportation cost per unit |
| \(M\) | computed | – | Sum of all \(d_j\) |

All indices and parameters are derived directly from the complete runtime records; no numerical values are enumerated in the model.