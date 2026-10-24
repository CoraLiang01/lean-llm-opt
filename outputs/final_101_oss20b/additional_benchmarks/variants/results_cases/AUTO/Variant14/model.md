## Mathematical Model

**Sets**  
\(I\) – set of candidate depots (indexed by the *Center* values in *facility_sites.csv*)  
\(J\) – set of service zones (indexed by the *Zone* values in *service_zones.csv*)

**Parameters**  
\(c_i\) – opening cost of depot \(i\) (from *OpeningCost* in *facility_sites.csv*)  
\(a_{ij}\) – 1 if depot \(i\) covers zone \(j\), 0 otherwise (derived from *CoveredDistricts*)

**Decision Variables**  
\(y_i \in \{0,1\}\) – 1 if depot \(i\) is opened, 0 otherwise

**Objective**  
\[
\min \sum_{i\in I} c_i\,y_i
\]

**Constraints**  
Coverage: for every zone \(j\in J\)
\[
\sum_{i\in I} a_{ij}\,y_i \;\ge\; 1
\]

Binary: for every depot \(i\in I\)
\[
y_i \in \{0,1\}
\]

---

## Data Mapping Rules

| Table ID | Source File | Columns Used | Mapping |
|----------|-------------|--------------|---------|
| `file_0_view_0` | `facility_sites.csv` | `Center`, `OpeningCost`, `CoveredDistricts` | For each record \(i\):<br>• \(c_i \gets\) value of *OpeningCost* (numeric).<br>• Split *CoveredDistricts* by `;` to obtain a list of zone identifiers. For each zone \(j\) in that list, set \(a_{ij}=1\); for all other \(j\in J\), set \(a_{ij}=0\). |
| `file_1_view_0` | `service_zones.csv` | `Zone` | Define set \(J\) as the collection of all *Zone* values. |

No other data are required. The model uses only the parameters and sets derived from these two tables.