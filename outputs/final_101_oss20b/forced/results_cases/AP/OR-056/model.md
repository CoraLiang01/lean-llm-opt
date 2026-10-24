**Mathematical Model**

Let  

- \(I\) be the set of display areas indexed by the values in column **DisplayID** of *capacity.csv*.  
- \(J\) be the set of vessel types indexed by the values in column **ProductName** of *products.csv*.  

Parameters  

\[
\begin{aligned}
\text{capacity}[i] &\;=\; \text{Capacity of display area } i, \\
\text{value}[j] &\;=\; \text{Value of vessel type } j, \\
\text{weight}[j] &\;=\; \text{Weight (size) of vessel type } j .
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;=\;\text{number of vessels of type } j \text{ placed in display area } i,
\qquad x_{ij}\in \mathbb{Z}_{\ge 0}\;\;\forall i\in I,\;j\in J .
\]

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J} \text{value}[j]\;x_{ij}
\]

**Constraints**

1. **Capacity limits** – the total weight of vessels in each display area must not exceed its capacity  

\[
\sum_{j\in J} \text{weight}[j]\;x_{ij}\;\le\;\text{capacity}[i] \quad \forall i\in I .
\]

2. **Non‑negativity and integrality** – already encoded in the variable definition.  

---

**Data Mapping**

| Parameter | Source Table | Source Column | Index Column |
|-----------|--------------|---------------|--------------|
| \(\text{capacity}[i]\) | `file_0_view_0` | `Capacity` | `DisplayID` |
| \(\text{value}[j]\) | `file_1_view_0` | `Value` | `ProductName` |
| \(\text{weight}[j]\) | `file_1_view_0` | `Weight` | `ProductName` |

All indices \(i\) and \(j\) are taken from the complete set of rows returned in the observation. No additional constraints or bounds are imposed beyond those stated.