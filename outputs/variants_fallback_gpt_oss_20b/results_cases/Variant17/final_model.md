**Symbolic Model**

Let  

* \(I=\{i\}\) – set of candidate locations (rows of *location_distance.csv*).  
* \(J=\{j\}\) – set of residential areas (rows of *area_demand.csv*).  

Parameters  

* \(d_j\) – demand of area \(j\) (from *area_demand.csv*).  
* \(c_{ij}\) – distance from location \(i\) to area \(j\) (from *location_distance.csv*).  
* \(P\) – required number of facilities to open (from *planning_parameters.csv*).  

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if location \(i\) is opened.  
* \(x_{ij} \in \{0,1\}\) – 1 if area \(j\) is assigned to location \(i\).  

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} d_j\,c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}=1 &&\forall j\in J \quad\text{(exact‑one assignment)}\\[4pt]
&x_{ij}\le y_i &&\forall i\in I,\; j\in J \quad\text{(linking)}\\[4pt]
&\sum_{i\in I} y_i = P &&\text{(exact number of facilities)}\\[4pt]
&x_{ij}\in\{0,1\},\; y_i\in\{0,1\} &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Key Column(s) | Value Column | Notes |
|-----------|--------------|---------------|--------------|-------|
| \(d_j\) | `file_0_view_0` | `Area` | `Demand` | Demand per area |
| \(c_{ij}\) | `file_1_view_0` | `Location` (row), `Area` (column) | distance value | Matrix of distances |
| \(P\) | `file_2_view_0` | `Parameter` = `NumberOfFacilitiesToOpen` | `Value` | Scalar parameter |

*Index sets*  
- \(I\) is the set of distinct `Location` values in `file_1_view_0`.  
- \(J\) is the set of distinct `Area` values in `file_0_view_0` (also the column headers of `file_1_view_0`).

These mappings bind each symbolic parameter to the exact source data without enumerating individual numeric entries.