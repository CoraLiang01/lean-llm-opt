**Mathematical Model**

Let  

- \(I=\{\,\text{CabinetID}\,\}\)  (index set of cabinets, from *file_0_view_0*)  
- \(J=\{\,\text{ProductName}\,\}\)  (index set of coffee products, from *file_1_view_0*)

Parameters  

\[
\begin{aligned}
\text{Capacity}_i &\quad\text{(weight capacity of cabinet }i), && i\in I \\
\text{Value}_j &\quad\text{(value per unit of product }j), && j\in J \\
\text{Weight}_j &\quad\text{(weight per unit of product }j), && j\in J
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Value}_j\,x_{ij}
\]

---

**Constraints**

1. **Cabinet capacity**

\[
\sum_{j\in J}\text{Weight}_j\,x_{ij}\;\le\;\text{Capacity}_i
\qquad \forall i\in I
\]

2. **Non‑negativity and integrality**

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| Capacity | `file_0_view_0` | `Capacity` |
| Value | `file_1_view_0` | `Value` |
| Weight | `file_1_view_0` | `Weight` |

The index sets \(I\) and \(J\) are derived from the `CabinetID` and `ProductName` columns of the respective tables. No numerical values are hard‑coded; all coefficients are linked to the source data via the mapping above.