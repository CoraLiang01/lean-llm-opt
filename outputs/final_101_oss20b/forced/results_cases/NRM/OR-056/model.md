**Mathematical Model**

Let  

- \(I\) be the set of display areas indexed by the values in column **DisplayID** of *file_0_view_0*.  
- \(J\) be the set of vessel types indexed by the values in column **ProductName** of *file_1_view_0*.  

Parameters  

\[
\begin{aligned}
\text{Capacity}_i &\;=\; \text{Capacity of display area } i \quad (i\in I)\\
\text{Value}_j &\;=\; \text{Value of vessel type } j \quad (j\in J)\\
\text{Weight}_j &\;=\; \text{Weight (size) of vessel type } j \quad (j\in J)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;=\;\text{number of vessels of type } j \text{ placed in display area } i,
\qquad i\in I,\; j\in J
\]

\[
x_{ij}\in \mathbb{Z}_{\ge 0}
\]

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J} \text{Value}_j \; x_{ij}
\]

**Constraints**

1. **Capacity constraints (per display area)**  

\[
\sum_{j\in J} \text{Weight}_j \; x_{ij}\;\le\; \text{Capacity}_i,
\qquad \forall\, i\in I
\]

2. **Non‑negativity**  

\[
x_{ij}\;\ge\;0,\qquad \forall\, i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` |
| Index \(i\) | `file_0_view_0` | `DisplayID` |
| Index \(j\) | `file_1_view_0` | `ProductName` |

All parameters are taken directly from the corresponding columns of the specified tables; no additional data or assumptions are introduced.