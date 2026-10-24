## Mathematical Model

**Index sets**  
\(I=\{\,i\mid i=\text{DisplayID from *capacity.csv*}\,\}\)  
\(J=\{\,j\mid j=\text{ProductName from *products.csv*}\,\}\)

**Parameters**  
\(\text{Cap}[i]\) – capacity of display area \(i\) (from *capacity.csv*)  
\(\text{Val}[j]\) – value of product \(j\) (from *products.csv*)  
\(\text{Wgt}[j]\) – weight of product \(j\) (from *products.csv*)

**Decision variables**  
\(x_{ij}\in\mathbb{Z}_{\ge 0}\) – number of vessels of type \(j\) placed in display area \(i\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Val}[j]\;x_{ij}
\]

**Constraints**  

Capacity of each display area  
\[
\sum_{j\in J}\text{Wgt}[j]\;x_{ij}\;\le\;\text{Cap}[i]\qquad \forall i\in I
\]

Non‑negativity and integrality  
\[
x_{ij}\;\ge\;0,\; x_{ij}\in\mathbb{Z}\qquad \forall i\in I,\; j\in J
\]

---

## Data Mapping

| Table ID | Column | Parameter | Source |
|----------|--------|-----------|--------|
| file_0_view_0 | DisplayID | \(i\) | *capacity.csv* |
| file_0_view_0 | Capacity | \(\text{Cap}[i]\) | *capacity.csv* |
| file_1_view_0 | ProductName | \(j\) | *products.csv* |
| file_1_view_0 | Value | \(\text{Val}[j]\) | *products.csv* |
| file_1_view_0 | Weight | \(\text{Wgt}[j]\) | *products.csv* |

All indices \(i\) and \(j\) are taken from the full set of rows returned in the observation; no additional filtering is applied. The model uses the exact values from the source tables as defined above.