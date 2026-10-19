**Mathematical Model**

Let  

* \(I=\{i\}\) – set of candidate clinics (rows of *clinic_distance.csv*).  
* \(J=\{j\}\) – set of neighborhoods (rows of *neighborhood_demand.csv*).  

Parameters  

* \(d_j\) – demand of neighborhood \(j\) (column *Demand*).  
* \(c_{ij}\) – travel distance from clinic \(i\) to neighborhood \(j\) (matrix entry).  
* \(P\) – required number of clinics to open (value of *NumberOfClinicsToOpen*).  

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if clinic \(i\) is opened.  
* \(x_{ij} \in \{0,1\}\) – 1 if neighborhood \(j\) is assigned to clinic \(i\).  

\[
\begin{aligned}
\min\;& \sum_{i\in I}\sum_{j\in J} d_j\,c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{i\in I} x_{ij}=1, &&\forall j\in J \quad\text{(each neighborhood assigned)}\\[2pt]
& x_{ij}\le y_i, &&\forall i\in I,\; j\in J \quad\text{(link)}\\[2pt]
& \sum_{i\in I} y_i = P, &&\text{(exactly \(P\) clinics)}\\[2pt]
& x_{ij}\in\{0,1\},\; y_i\in\{0,1\}, &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Columns | Mapping Type |
|-----------|--------------|----------------|--------------|
| \(d_j\) | `file_0_view_0` | `Neighborhood` (id), `Demand` | row‑to‑parameter |
| \(c_{ij}\) | `file_1_view_0` | `Clinic` (row id), `N1`–`N10` (column ids) | matrix |
| \(P\) | `file_2_view_0` | `Parameter` (must equal `"NumberOfClinicsToOpen"`), `Value` | single‑value |

*Index sets*  
* \(I\) = set of values in column `Clinic` of `file_1_view_0`.  
* \(J\) = set of values in column `Neighborhood` of `file_0_view_0`.  

All mappings use exact identifier matching; no additional filtering beyond the specified conditions.