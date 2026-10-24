**Mathematical Model**

Let  

* \(I=\{i\}\) be the set of candidate facilities (from *facility_distance.csv*).  
* \(J=\{j\}\) be the set of neighborhoods (from *neighborhood_demand.csv*).  

Parameters  

* \(D_j\) – demand of neighborhood \(j\) (from *neighborhood_demand.csv*).  
* \(d_{ij}\) – distance from facility \(i\) to neighborhood \(j\) (from *facility_distance.csv*).  
* \(P\) – required number of facilities to open (from *planning_parameters.csv*).  

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if facility \(i\) is opened.  
* \(x_{ij} \in \{0,1\}\) – 1 if neighborhood \(j\) is assigned to facility \(i\).  

---

### Objective
\[
\min \;\sum_{i\in I}\sum_{j\in J} D_j\, d_{ij}\, x_{ij}
\]

### Constraints
1. **Exactly one facility per neighborhood**  
   \[
   \sum_{i\in I} x_{ij}=1,\qquad \forall j\in J
   \]

2. **Exactly \(P\) facilities opened**  
   \[
   \sum_{i\in I} y_i = P
   \]

3. **Assignment only to opened facilities**  
   \[
   x_{ij}\le y_i,\qquad \forall i\in I,\; j\in J
   \]

4. **Binary domains**  
   \[
   y_i\in\{0,1\},\; x_{ij}\in\{0,1\},\qquad \forall i\in I,\; j\in J
   \]

---

## Data Mapping

| Symbol | Source Table | Column(s) | Notes |
|--------|---------------|-----------|-------|
| \(I\) | `file_1_view_0` | `Facility` | Set of facility identifiers |
| \(J\) | `file_0_view_0` | `Neighborhood` | Set of neighborhood identifiers |
| \(D_j\) | `file_0_view_0` | `Demand` | Demand per neighborhood |
| \(d_{ij}\) | `file_1_view_0` | `N1`–`N10` | Distance matrix (row = facility, column = neighborhood) |
| \(P\) | `file_2_view_0` | `Value` where `Parameter = "NumberOfFacilitiesToOpen"` | Number of facilities to open |

All parameters are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.