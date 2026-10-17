**Mathematical Model**

Let  

* \(I=\{i\}\) – set of display areas (indices given by *DisplayID* in **file_0_view_0**)  
* \(J=\{j\}\) – set of boat types (indices given by *ProductName* in **file_1_view_0**)  

Parameters  

* \(\text{Cap}_i\) – capacity of display area \(i\) (units of size)  
* \(\text{Val}_j\) – monetary value of one unit of boat type \(j\)  
* \(\text{Wgt}_j\) – size (weight) of one unit of boat type \(j\)

Decision variables  

* \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of boat type \(j\) placed in display area \(i\)

---

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} \text{Val}_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J} \text{Wgt}_j\,x_{ij}\;\le\;\text{Cap}_i,
\qquad \forall\, i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0},
\qquad \forall\, i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Index Column |
|-----------|--------------|---------------|--------------|
| \(\text{Cap}_i\) | `file_0_view_0` | `Capacity` | `DisplayID` |
| \(\text{Val}_j\) | `file_1_view_0` | `Value` | `ProductName` |
| \(\text{Wgt}_j\) | `file_1_view_0` | `Weight` | `ProductName` |

All other model components are defined symbolically; no numerical values are enumerated.