**Mathematical Model**

Let  

- \(I=\{\,i\mid i=\text{StorageID in file_0_view_0}\,\}\)  
- \(J=\{\,j\mid j=\text{ProductName in file_1_view_0}\,\}\)

Parameters  

\[
\begin{aligned}
\text{Capacity}_i &\;=\; \text{Capacity of storage }i \quad (i\in I)\\
\text{Value}_j     &\;=\; \text{Value of product }j \quad (j\in J)\\
\text{Weight}_j    &\;=\; \text{Weight (size) of product }j \quad (j\in J)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;\;=\;\;\text{number of units of product }j\text{ placed in storage }i,
\qquad x_{ij}\in\mathbb{Z}_{\ge 0}\;\;(i\in I,\;j\in J)
\]

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Value}_j\,x_{ij}
\]

**Constraints**

1. **Capacity constraints (per storage area)**  

\[
\sum_{j\in J}\text{Weight}_j\,x_{ij}\;\le\;\text{Capacity}_i,
\qquad \forall\,i\in I
\]

2. **Non‑negativity and integrality**  

\[
x_{ij}\in\mathbb{Z}_{\ge 0},\qquad \forall\,i\in I,\;j\in J
\]

---

### Data Mapping

| Parameter / Variable | Table ID          | Column(s) Used |
|----------------------|-------------------|----------------|
| \(\text{Capacity}_i\) | `file_0_view_0`   | `StorageID` (index), `Capacity` (value) |
| \(\text{Value}_j\)    | `file_1_view_0`   | `ProductName` (index), `Value` (value) |
| \(\text{Weight}_j\)   | `file_1_view_0`   | `ProductName` (index), `Weight` (value) |
| \(x_{ij}\)            | –                 | – (decision variable) |

All indices \(i\) and \(j\) are taken directly from the respective tables; no additional enumeration or sorting is performed.