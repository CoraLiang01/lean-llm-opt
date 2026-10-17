**Mathematical Model**

Let  

* \(I=\{S_1,\dots ,S_{12}\}\) – set of suppliers  
* \(J=\{C_1,\dots ,C_{12}\}\) – set of supermarkets  

Parameters  

* \(f_i\) – fixed opening cost of supplier \(i\)  
* \(d_j\) – demand (units) of supermarket \(j\)  
* \(t_{ij}\) – transportation cost per unit from supplier \(i\) to supermarket \(j\)

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to supermarket \(j\)

---

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij}= d_j \qquad &&\forall j\in J \quad\text{(demand satisfaction)}\\[4pt]
& x_{ij}\;\le\; d_j\,y_i \qquad &&\forall i\in I,\; j\in J \quad\text{(link to opening)}\\[4pt]
& y_i\in\{0,1\} \qquad &&\forall i\in I\\
& x_{ij}\ge 0 \qquad &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column | Notes |
|-----------|--------------|------------|--------------|-------|
| \(f_i\) | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | Supplier identifier |
| \(d_j\) | `file_0_view_0` | `customer` | `demand` | Supermarket identifier |
| \(t_{ij}\) | `file_2_view_0` | `Unnamed: 0` (supplier) | `C1`–`C12` (supermarkets) | Transportation cost matrix |

All indices \(i\) and \(j\) are taken from the distinct values appearing in the corresponding key columns of the source tables. No numerical values are enumerated in the model; they are referenced through the mapping above.