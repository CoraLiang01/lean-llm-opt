**Mathematical Model**

Let  

- \(F=\{A1,\dots ,A15\}\) (Factory sites)  
- \(D=\{B1,\dots ,B8\}\) (Distribution centers)

Parameters  

| Symbol | Definition | Source |
|--------|------------|--------|
| \(c_f\) | Fixed construction cost of factory \(f\) | `file_0_view_0` column **FixedCost** |
| \(K_f\) | Production capacity of factory \(f\) | `file_0_view_0` column **Capacity** |
| \(d_d\) | Demand at destination \(d\) | `file_2_view_0` column **Demand** |
| \(s_{f d}\) | Variable shipping cost per unit from \(f\) to \(d\) | `file_1_view_0` matrix entry (row \(f\), column \(d\)) |

Decision variables  

- \(y_f \in \{0,1\}\) 1 if factory \(f\) is built, 0 otherwise  
- \(x_{f d}\ge 0\) units shipped from \(f\) to \(d\)

Objective (minimise total cost)

\[
\min \;\sum_{f\in F} c_f\,y_f \;+\; \sum_{f\in F}\sum_{d\in D} s_{f d}\,x_{f d}
\]

Supply‑capacity constraints (only if factory is built)

\[
\sum_{d\in D} x_{f d}\;\le\; K_f\,y_f \qquad \forall f\in F
\]

Demand‑satisfaction constraints

\[
\sum_{f\in F} x_{f d}\;\ge\; d_d \qquad \forall d\in D
\]

Non‑negativity and integrality

\[
x_{f d}\ge 0 \quad \forall f\in F,\; d\in D,\qquad
y_f\in\{0,1\}\quad \forall f\in F
\]

---

**Data Mapping**

| Parameter | Table ID | Column(s) | Notes |
|-----------|----------|-----------|-------|
| \(c_f\) | `file_0_view_0` | `FixedCost` | one value per facility |
| \(K_f\) | `file_0_view_0` | `Capacity` | one value per facility |
| \(d_d\) | `file_2_view_0` | `Demand` | one value per destination |
| \(s_{f d}\) | `file_1_view_0` | matrix of shipping costs | row = facility, column = destination |

All index sets \(F\) and \(D\) are taken from the complete rows of the respective tables; no additional filtering is applied. The model uses the exact values from these tables and does not introduce any new parameters or constraints beyond those required by the problem statement.