## Mathematical Model

Let  

* \(I=\{1,\dots ,14\}\) – set of display areas (index taken from **capacity.csv**).  
* \(J=\{1,\dots ,20\}\) – set of boat types (index taken from **products.csv**).  

Parameters  

* \(c_i\) – capacity of display area \(i\) (units of weight).  
* \(w_j\) – weight of boat type \(j\).  
* \(v_j\) – value of boat type \(j\).  

Decision variables  

* \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of boat type \(j\) placed in display area \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J} w_j\,x_{ij}\;\le\; c_i , \qquad \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, \qquad \forall i\in I,\; j\in J .
\end{aligned}
\]

---

## Data Mapping

| Parameter | Source | Table ID | Column(s) | Mapping |
|-----------|--------|----------|-----------|---------|
| \(c_i\) | capacity.csv | `file_0_view_0` | `Capacity` | \(c_{DisplayID}\) |
| \(w_j\) | products.csv | `file_1_view_0` | `Weight` | \(w_{ProductName}\) |
| \(v_j\) | products.csv | `file_1_view_0` | `Value` | \(v_{ProductName}\) |

*Index sets*  
- \(I\) is the set of all `DisplayID` values in `file_0_view_0`.  
- \(J\) is the set of all `ProductName` values in `file_1_view_0`.

All parameters are taken directly from the corresponding columns; no additional processing or filtering is applied. The model uses the full set of rows returned in each file.