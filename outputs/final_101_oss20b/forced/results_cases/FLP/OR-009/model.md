**Mathematical Model**

Let  

- \(I=\{i\mid i\text{ is a supplier in }file\_1\_view\_0\}\)  
- \(J=\{j\mid j\text{ is a customer in }file\_0\_view\_0\}\)

Decision variables  

\[
x_{ij}\ge 0 \qquad \forall i\in I,\; j\in J
\]

Parameters  

\[
\begin{aligned}
d_j &\;=\;\text{demand of customer }j \\
s_i &\;=\;\text{production capacity of supplier }i \\
c_{ij}&\;=\;\text{transportation cost per unit from }i\text{ to }j
\end{aligned}
\]

Objective  

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

\[
\begin{aligned}
\text{(Demand)}\quad &\sum_{i\in I} x_{ij}=d_j &&\forall j\in J\\[4pt]
\text{(Capacity)}\quad &\sum_{j\in J} x_{ij}\le s_i &&\forall i\in I
\end{aligned}
\]

All variables are continuous and non‑negative.

---

**Data Mapping**

| Parameter | Table ID | Row index column | Column name |
|-----------|----------|------------------|-------------|
| \(d_j\) | `file_0_view_0` | `customer` | `demand` |
| \(s_i\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` | `C1`, `C2`, `C3`, `C4` (corresponding to \(j\)) |

The index sets \(I\) and \(J\) are taken directly from the row identifiers of the respective tables. No additional constraints or parameters are introduced beyond those specified.