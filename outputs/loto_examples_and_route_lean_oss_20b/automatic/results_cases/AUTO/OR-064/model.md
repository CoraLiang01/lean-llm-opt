**Mathematical Model**

Sets  
\[
S=\{S_1,\dots ,S_{24}\},\qquad  
C=\{C_1,\dots ,C_{25}\}
\]

Parameters  
\[
\begin{aligned}
d_j &\;=\;\text{demand of customer }j\quad (j\in C)\\
f_i &\;=\;\text{fixed cost of supplier }i\quad (i\in S)\\
t_{ij}&\;=\;\text{transportation cost per unit from supplier }i\text{ to customer }j\\
M &\;=\;\sum_{j\in C} d_j \quad(\text{a sufficiently large constant})
\end{aligned}
\]

Decision variables  
\[
\begin{aligned}
y_i &\in\{0,1\}\quad(i\in S) &&\text{(supplier open)}\\
x_{ij}&\ge 0 \quad(i\in S,\;j\in C) &&\text{(units shipped)}
\end{aligned}
\]

Objective  
\[
\min Z=\sum_{i\in S} f_i\,y_i+\sum_{i\in S}\sum_{j\in C} t_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
\[
\sum_{i\in S} x_{ij}=d_j \qquad \forall j\in C
\]

2. **Supplier activation linking**  
\[
x_{ij}\le M\,y_i \qquad \forall i\in S,\;j\in C
\]

3. **Non‑negativity**  
\[
x_{ij}\ge 0 \qquad \forall i\in S,\;j\in C
\]

4. **Binary**  
\[
y_i\in\{0,1\}\qquad \forall i\in S
\]

---

### Data Mapping

| Parameter | Source Table | Row Identifier | Column Identifier |
|-----------|--------------|----------------|-------------------|
| \(d_j\) | `file_0_view_0` | `customer` | `demand` |
| \(f_i\) | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` |
| \(t_{ij}\) | `file_2_view_0` | `Unnamed: 0` | `C1`–`C25` (column matching customer \(j\)) |
| \(M\) | computed | – | – |

All indices \(i\) and \(j\) are taken from the full sets returned by the respective tables; no additional filtering is applied. The model uses the exact numeric values from the tables, and the big‑\(M\) constant is defined as the sum of all customer demands.