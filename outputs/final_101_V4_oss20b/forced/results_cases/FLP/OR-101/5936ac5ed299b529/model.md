**Mathematical Model**

Let  

- \(I=\{A,B,C,D,E,F,G,H,I,J\}\) (devices)  
- \(J=\{P1,P2,\dots ,P111\}\) (products)

Decision variables  

\[
x_{ij}\;\ge 0 \qquad \forall i\in I,\; j\in J
\]

Parameters  

\[
\begin{aligned}
p_j &\;=\;\text{unit profit of product }j,\\
t_{ij}&\;=\;\text{processing time of product }j\text{ on device }i,\\
c_i &\;=\;\text{monthly capacity of device }i .
\end{aligned}
\]

**Objective**

\[
\max \;\sum_{j\in J} p_j \sum_{i\in I} x_{ij}
\]

**Constraints**

\[
\sum_{j\in J} t_{ij}\,x_{ij}\;\le\; c_i \qquad \forall i\in I
\]

\[
x_{ij}\;\ge\;0 \qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Mapping Rule |
|-----------|--------------|--------------|
| \(t_{ij}\) | `file_0_view_0` | Row identified by `Device` = \(i\); column identified by `Product` = \(j\). |
| \(c_i\) | `file_1_view_0` | Row identified by `Device` = \(i\); column `Monthly_Capacity`. |
| \(p_j\) | `file_2_view_0` | Row identified by `Product` = \(j\); column `Unit_Profit`. |

All numeric values are parsed as floating‑point numbers. No other data or constraints are required.