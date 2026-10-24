**Model**

Let  

* \(I=\{S1,S2,S3,S4,S5,S6\}\) (suppliers)  
* \(J=\{C1,C2,C3,C4,C5,C6\}\) (stores)

Parameters  

* \(f_i\) fixed opening cost of supplier \(i\)  \(i\in I\)  
* \(c_{ij}\) transportation cost per unit from supplier \(i\) to store \(j\) \(i\in I,\;j\in J\)  
* \(d_j\) demand of store \(j\) \(j\in J\)

Decision variables  

* \(y_i\in\{0,1\}\) 1 if supplier \(i\) is opened, 0 otherwise \(i\in I\)  
* \(x_{ij}\ge 0\) quantity shipped from supplier \(i\) to store \(j\) \(i\in I,\;j\in J\)

Let \(M=\sum_{j\in J} d_j\) (any sufficiently large number).

\[
\begin{aligned}
\min\;& Z = \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij} = d_j \qquad \forall j\in J \\[4pt]
& x_{ij} \le M\,y_i \qquad \forall i\in I,\;j\in J \\[4pt]
& x_{ij}\ge 0 \qquad \forall i\in I,\;j\in J \\[4pt]
& y_i\in\{0,1\} \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Row‑ID column | Column(s) | Notes |
|-----------|----------|---------------|-----------|-------|
| \(f_i\) | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | Supplier identifier in `Unnamed: 0` |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` | `C1`–`C6` | Row supplier, column store |
| \(d_j\) | `file_0_view_0` | `customer` | `demand` | Store identifier in `customer` |

All indices \(I\) and \(J\) are taken from the full set of rows in the corresponding tables; no additional filtering is applied. The model uses the exact raw identifiers (`S1`–`S6`, `C1`–`C6`) as index values.