## Mathematical Model

**Sets**  
\(I=\{\,\text{SectionID}\mid \text{SectionID in file\_0\_view\_0}\,\}\)  
\(J=\{\,\text{ProductName}\mid \text{ProductName in file\_1\_view\_0}\,\}\)

**Parameters**  
\[
\begin{aligned}
\text{Capacity}_i &:= \text{Capacity from file\_0\_view\_0 where SectionID}=i, \qquad i\in I\\
\text{Value}_j &:= \text{Value from file\_1\_view\_0 where ProductName}=j, \qquad j\in J\\
\text{Weight}_j &:= \text{Weight from file\_1\_view\_0 where ProductName}=j, \qquad j\in J
\end{aligned}
\]

**Decision Variables**  
\(x_{ij}\in \mathbb{Z}_{+}\) units of product \(j\) stocked in section \(i\), \(i\in I,\;j\in J\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Value}_j\,x_{ij}
\]

**Constraints**  

1. **Section capacity**  
\[
\sum_{j\in J}\text{Weight}_j\,x_{ij}\;\le\;\text{Capacity}_i
\quad\forall\,i\in I
\]

2. **Non‑negativity & integrality**  
\[
x_{ij}\;\ge\;0,\qquad x_{ij}\in\mathbb{Z}
\quad\forall\,i\in I,\;j\in J
\]

---

## Data Mapping

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| `Capacity_i` | `file_0_view_0` | `Capacity` | For each `SectionID` \(i\) in `file_0_view_0`, assign `Capacity_i` = `Capacity` |
| `Value_j` | `file_1_view_0` | `Value` | For each `ProductName` \(j\) in `file_1_view_0`, assign `Value_j` = `Value` |
| `Weight_j` | `file_1_view_0` | `Weight` | For each `ProductName` \(j\) in `file_1_view_0`, assign `Weight_j` = `Weight` |

Indices `i` and `j` are taken directly from the `SectionID` and `ProductName` columns of the respective tables. No additional filtering or aggregation is applied.