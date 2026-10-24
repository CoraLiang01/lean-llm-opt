**Mathematical Model**

Let  

- \(I\) be the set of display areas (indices from the *capacity* table).  
- \(J\) be the set of boat types (indices from the *products* table).  

Parameters  

- \(c_i\) – capacity of display area \(i\).  
- \(v_j\) – value of one unit of boat type \(j\).  
- \(w_j\) – size (weight) of one unit of boat type \(j\).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of boat type \(j\) placed in display area \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J} w_j\,x_{ij}\;\le\; c_i \qquad \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) |
|-----------|--------------|-----------|
| \(c_i\)   | `file_0_view_0` | `DisplayID`, `Capacity` |
| \(v_j\)   | `file_1_view_0` | `ProductName`, `Value` |
| \(w_j\)   | `file_1_view_0` | `ProductName`, `Weight` |

*Index sets*  
- \(I = \{\,\text{DisplayID}\,\}\) from `file_0_view_0`.  
- \(J = \{\,\text{ProductName}\,\}\) from `file_1_view_0`.  

All mappings use the exact identifiers and column names as they appear in the source tables. No numerical values are enumerated in the model.