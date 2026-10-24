**Mathematical model**

Let  

\[
I=\{C1,\dots ,C15\}, \qquad J=\{SC1,\dots ,SC10\}
\]

Parameters  

\[
\begin{aligned}
f_j &\;=\;\text{fixed opening cost of centre }j, && j\in J\\
c_{ij}&\;=\;\text{service cost of assigning customer }i\text{ to centre }j, && i\in I,\;j\in J
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_j &\in\{0,1\} &&\text{(centre }j\text{ is opened)}\\
x_{ij}&\in\{0,1\} &&\text{(customer }i\text{ is served by centre }j)
\end{aligned}
\]

Objective  

\[
\min \;\sum_{j\in J} f_j\,y_j \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Customer assignment**  
\[
\sum_{j\in J} x_{ij}=1 \qquad \forall i\in I
\]

2. **Open‑centre requirement**  
\[
x_{ij}\le y_j \qquad \forall i\in I,\; j\in J
\]

3. **Capacity limit**  
\[
\sum_{i\in I} x_{ij}\le 4 \qquad \forall j\in J
\]

4. **Variable domains**  
\[
x_{ij}\in\{0,1\},\; y_j\in\{0,1\}\qquad \forall i\in I,\; j\in J
\]

---

**Data mapping**

| Parameter | Source table | Row key | Column key | Mapping rule |
|-----------|--------------|---------|------------|--------------|
| \(f_j\) | `file_1_view_0` | `Service Center` | `Fixed Opening Cost` | \(f_j =\) value in row where `Service Center` = \(j\) |
| \(c_{ij}\) | `file_0_view_0` | `Customer` | `SCj` | \(c_{ij} =\) value in row where `Customer` = \(i\) and column `SCj` |

All numeric values are taken directly from the corresponding CSV files; no enumeration of individual entries is shown.