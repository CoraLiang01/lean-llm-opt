**Mathematical Model**

Let  

* \(I\) – set of shelves (indexed by `ShelfID`),  
* \(J\) – set of products (indexed by `ProductName`).

Parameters  

\[
\begin{aligned}
\text{Capacity}_i &\quad (i\in I) \\
\text{Value}_j &\quad (j\in J) \\
\text{Weight}_j &\quad (j\in J)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad (i\in I,\; j\in J)
\]

---

**Objective**

\[
\max \;\; \sum_{i\in I}\sum_{j\in J} \text{Value}_j \; x_{ij}
\]

---

**Constraints**

1. **Shelf‑capacity (weight) limits**

\[
\sum_{j\in J} \text{Weight}_j \; x_{ij}\;\le\; \text{Capacity}_i
\qquad \forall\, i\in I
\]

2. **Non‑negativity (implicit in integer domain)**  

\[
x_{ij}\;\ge\;0 \qquad \forall\, i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` |
| Index \(i\) | `file_0_view_0` | `ShelfID` |
| Index \(j\) | `file_1_view_0` | `ProductName` |

All parameter values are taken directly from the corresponding columns of the specified tables; no additional data or assumptions are introduced.