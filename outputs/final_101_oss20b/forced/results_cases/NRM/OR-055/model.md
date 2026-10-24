**Mathematical Model**

Let  

- \(I\) – set of display areas (indexed by the values in *DisplayID*).  
- \(J\) – set of boat types (indexed by the values in *ProductName*).  

Parameters  

- \(c_i\) – capacity of display area \(i\) (units of size).  
- \(v_j\) – monetary value of one unit of boat type \(j\).  
- \(w_j\) – size (weight) of one unit of boat type \(j\).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of boat type \(j\) placed in display area \(i\).  

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{j\in J} w_j\,x_{ij}\;\le\; c_i , && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, && \forall i\in I,\; j\in J .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(c_i\)   | `file_0_view_0` | `Capacity` |
| \(v_j\)   | `file_1_view_0` | `Value` |
| \(w_j\)   | `file_1_view_0` | `Weight` |

The index sets \(I\) and \(J\) are derived from the distinct values of `DisplayID` and `ProductName` in the respective tables. No additional data or constraints are introduced beyond those specified in the query.