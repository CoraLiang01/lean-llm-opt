**Mathematical Model**

Let  

- \(I=\{S1,\dots ,S8\}\) suppliers  
- \(J=\{C1,\dots ,C9\}\) dealerships  

Decision variables  

\[
\begin{aligned}
y_i &\in \{0,1\} &&\text{(supplier \(i\) is opened)}\\
x_{ij} &\ge 0 &&\text{(vehicles supplied from supplier \(i\) to dealership \(j\))}
\end{aligned}
\]

Parameters  

\[
\begin{aligned}
f_i &\ge 0 &&\text{fixed cost of opening supplier }i\\
c_{ij} &\ge 0 &&\text{transportation cost per vehicle from }i\text{ to }j\\
d_j &\ge 0 &&\text{demand of dealership }j
\end{aligned}
\]

Objective  

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;\ge\; d_j \qquad \forall j\in J
   \]

2. **Supplier activation** (big‑\(M\) with \(M_j=d_j\))  
   \[
   x_{ij} \;\le\; d_j\,y_i \qquad \forall i\in I,\; j\in J
   \]

3. **Variable domains**  
   \[
   y_i\in\{0,1\}\;\;\forall i\in I,\qquad
   x_{ij}\ge 0\;\;\forall i\in I,\;j\in J
   \]

---

### Data Mapping

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | `i` ↔ `Unnamed: 0` |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C9` | `i` ↔ `Unnamed: 0`, `j` ↔ column header |
| \(d_j\) | `file_0_view_0` | `demand` | `j` ↔ `customer` |

All indices \(i\) and \(j\) are taken from the complete set of rows/columns returned in the observation; no additional filtering is applied. The model uses the exact numeric values from the tables as provided.