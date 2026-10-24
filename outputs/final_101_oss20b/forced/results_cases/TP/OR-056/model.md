**Mathematical Model**

Let  

- \(I=\{i\mid\text{DisplayID in } \text{capacity.csv}\}\)  
- \(J=\{j\mid\text{ProductName in } \text{products.csv}\}\)

Parameters  

\[
\begin{aligned}
\text{cap}_i &:= \text{Capacity of display } i, && i\in I\\
\text{val}_j &:= \text{Value of product } j, && j\in J\\
\text{wt}_j  &:= \text{Weight of product } j, && j\in J
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{+}\quad\forall i\in I,\; j\in J
\]

Objective  

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{val}_j\,x_{ij}
\]

Capacity constraints  

\[
\sum_{j\in J}\text{wt}_j\,x_{ij}\;\le\;\text{cap}_i
\qquad\forall i\in I
\]

Non‑negativity and integrality are implicit in the variable definition.

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column | Mapping |
|-----------|--------------|---------------|---------|
| \(\text{cap}_i\) | `file_0_view_0` | `Capacity` | `i` ← `DisplayID` |
| \(\text{val}_j\) | `file_1_view_0` | `Value` | `j` ← `ProductName` |
| \(\text{wt}_j\)  | `file_1_view_0` | `Weight` | `j` ← `ProductName` |

All indices \(i\) and \(j\) are taken directly from the respective tables; no additional filtering or aggregation is applied.