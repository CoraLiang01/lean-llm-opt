**Mathematical Model**

Let  

* \(I=\{S_1,\dots ,S_8\}\) – set of suppliers  
* \(J=\{C_1,\dots ,C_9\}\) – set of dealerships  

Parameters  

\[
\begin{aligned}
d_j &\quad\text{demand of dealership }j\in J,\\
f_i &\quad\text{fixed cost of opening supplier }i\in I,\\
c_{ij}&\quad\text{transportation cost per vehicle from supplier }i\text{ to dealership }j,\\
M &= \sum_{j\in J} d_j \quad\text{(big‑\(M\) upper bound)} .
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
x_{ij}&\ge 0 &&\text{vehicles shipped from }i\text{ to }j,\\
y_i &\in\{0,1\} &&\text{supplier }i\text{ is opened}.
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j \qquad \forall j\in J
   \]

2. **Supplier activation (big‑\(M\) formulation)**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; M\,y_i \qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_i\in\{0,1\}\qquad \forall i\in I,\; j\in J
   \]

---

### Data Mapping

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(d_j\) | `file_0_view_0` | `demand` | `customer` column gives \(j\) |
| \(f_i\) | `file_1_view_0` | `fixed_costs` | `Unnamed: 0` column gives \(i\) |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C9` | `Unnamed: 0` column gives \(i\); column name `Ck` gives \(j=C_k\) |
| \(M\) | computed | – | \(M=\sum_{j\in J} d_j\) |

All indices \(i\) and \(j\) are taken from the full sets of suppliers and dealerships present in the respective tables. No additional capacity limits are imposed beyond the big‑\(M\) bound.