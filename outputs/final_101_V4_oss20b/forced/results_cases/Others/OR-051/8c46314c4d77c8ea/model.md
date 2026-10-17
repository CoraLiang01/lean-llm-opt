## Mathematical Model

**Sets**  
\(I=\{1,\dots ,10\}\) (cabinet indices)  
\(J=\{1,\dots ,18\}\) (product indices)

**Parameters**  
\(\text{Cap}_i\) – capacity of cabinet \(i\) (kg)  \(i\in I\)  
\(\text{Val}_j\) – value of one unit of product \(j\) (yuan) \(j\in J\)  
\(\text{Wgt}_j\) – weight of one unit of product \(j\) (kg) \(j\in J\)

**Decision variables**  
\(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed in cabinet \(i\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Val}_j\,x_{ij}
\]

**Constraints**  

Capacity limits for each cabinet  
\[
\sum_{j\in J}\text{Wgt}_j\,x_{ij}\;\le\;\text{Cap}_i
\qquad \forall\,i\in I
\]

All variables are non‑negative integers (already enforced by the domain).

---

## Data Mapping

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Cap}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{Val}_j\) | `file_1_view_0` | `Value` |
| \(\text{Wgt}_j\) | `file_1_view_0` | `Weight` |

Indices \(i\) and \(j\) are taken from the full set of rows in the respective tables (10 cabinets, 18 products). No additional filtering or aggregation is applied.