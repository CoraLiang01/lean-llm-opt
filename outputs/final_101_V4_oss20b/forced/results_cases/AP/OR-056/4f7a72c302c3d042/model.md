**Mathematical Model**

Let  

- \(I\) be the set of display areas.  
- \(J\) be the set of vessel types.  

Parameters  

\[
\begin{aligned}
c_i &\quad\text{(capacity of display area } i), && i\in I\\
v_j &\quad\text{(value of vessel type } j), && j\in J\\
w_j &\quad\text{(weight/size of vessel type } j), && j\in J
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;\;=\;\;\text{number of vessels of type } j \text{ placed in display area } i,
\qquad i\in I,\; j\in J
\]

\[
x_{ij}\in \mathbb{Z}_{\ge 0}
\]

---

**Objective**

\[
\max \;\; \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}
\]

---

**Constraints**

1. **Capacity limits**

\[
\sum_{j\in J} w_j\,x_{ij}\;\le\; c_i,
\qquad \forall\, i\in I
\]

2. **Non‑negativity / integrality**

\[
x_{ij}\in \mathbb{Z}_{\ge 0},
\qquad \forall\, i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(c_i\)   | `file_0_view_0` | `Capacity` |
| \(v_j\)   | `file_1_view_0` | `Value` |
| \(w_j\)   | `file_1_view_0` | `Weight` |

*Index sets*  
- \(I\) is the set of all `DisplayID` values in `file_0_view_0`.  
- \(J\) is the set of all `ProductName` values in `file_1_view_0`.