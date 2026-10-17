## Mathematical Model

**Index sets**  
\(I=\{\,i\mid i=\text{DisplayID from } \text{file\_0\_view\_0}\,\}\)  
\(J=\{\,j\mid j=\text{ProductName from } \text{file\_1\_view\_0}\,\}\)

**Parameters**  
\(\text{Cap}[i]   =\) Capacity of display area \(i\) (from column *Capacity* of *file_0_view_0*)  
\(\text{Val}[j]   =\) Value of product \(j\) (from column *Value* of *file_1_view_0*)  
\(\text{Wgt}[j]   =\) Weight of product \(j\) (from column *Weight* of *file_1_view_0*)

**Decision variables**  
\(x_{ij}\in\mathbb{Z}_{\ge 0}\) – number of vessels of type \(j\) placed in display area \(i\).

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Val}[j]\;x_{ij}
\]

**Constraints**  

Capacity of each display area  
\[
\sum_{j\in J}\text{Wgt}[j]\;x_{ij}\;\le\;\text{Cap}[i]\qquad \forall\,i\in I
\]

Non‑negativity and integrality  
\[
x_{ij}\;\ge\;0,\;\;x_{ij}\in\mathbb{Z}\qquad \forall\,i\in I,\;j\in J
\]

---

## Data Mapping

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Cap}[i]\) | `file_0_view_0` | `Capacity` |
| \(\text{Val}[j]\) | `file_1_view_0` | `Value` |
| \(\text{Wgt}[j]\) | `file_1_view_0` | `Weight` |

| Index Set | Source Table | Column |
|-----------|--------------|--------|
| \(I\) | `file_0_view_0` | `DisplayID` |
| \(J\) | `file_1_view_0` | `ProductName` |

All values are taken directly from the corresponding CSV files; no additional processing or enumeration is required.