**Mathematical Model**

Let  

- \(I=\{\,i\mid i=\text{DisplayID from *capacity.csv*}\,\}\)  
- \(J=\{\,j\mid j=\text{ProductName from *products.csv*}\,\}\)

Parameters  

\[
\begin{aligned}
\text{Capacity}_i &\;=\; \text{Capacity of display }i \quad (i\in I)\\
\text{Value}_j     &\;=\; \text{Value of product }j \quad (j\in J)\\
\text{Weight}_j    &\;=\; \text{Weight (size) of product }j \quad (j\in J)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;\in\;\mathbb{Z}_{\ge 0}\qquad (i\in I,\;j\in J)
\]

---

### Objective  

\[
\max \;\sum_{i\in I}\;\sum_{j\in J}\;\text{Value}_j \; x_{ij}
\]

### Constraints  

1. **Display‑capacity constraints**  

\[
\sum_{j\in J}\;\text{Weight}_j \; x_{ij}\;\le\;\text{Capacity}_i
\qquad \forall\, i\in I
\]

2. **Non‑negativity and integrality**  

\[
x_{ij}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall\, i\in I,\;j\in J
\]

---

## Data Mapping

| Parameter | Source Table | Column | Index Key |
|-----------|--------------|--------|-----------|
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` | `DisplayID` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` | `ProductName` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` | `ProductName` |

All other model components are defined symbolically; no numerical values are hard‑coded.