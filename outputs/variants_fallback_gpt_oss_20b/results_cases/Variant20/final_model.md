## Mathematical Model

**Sets**  
\(I\) – set of candidate sensor sites (indexed by *i*).  
\(J\) – set of monitoring zones (indexed by *j*).

**Parameters**  
\(c_i\) – installation cost of sensor at site *i*.  
\(a_{ij}\in\{0,1\}\) – 1 if site *i* covers zone *j*, 0 otherwise.

**Decision Variables**  
\(y_i\in\{0,1\}\) – 1 if a sensor is installed at site *i*, 0 otherwise.

**Objective**  
\[
\min \;\sum_{i\in I} c_i\,y_i
\]

**Constraints**  
Coverage: for every zone \(j\in J\)
\[
\sum_{i\in I} a_{ij}\,y_i \;\ge\; 1
\]

Binary: for every site \(i\in I\)
\[
y_i \in \{0,1\}
\]

---

## Data Mapping

| Table ID | Source Columns | Mapping to Model |
|----------|----------------|------------------|
| `file_0_view_0` | `Center` → \(i\) (site identifier) <br> `OpeningCost` → \(c_i\) <br> `CoveredDistricts` → list of zones covered by site *i* | For each record *i*: set \(a_{ij}=1\) if zone *j* appears in the semicolon‑separated list `CoveredDistricts`; otherwise \(a_{ij}=0\). |
| `file_1_view_0` | `Zone` → \(j\) (zone identifier) | All zones in this table form the index set \(J\). |

No additional parameters or constraints are introduced beyond those specified in the query. The model uses only the data retrieved from the two tables and respects the binary nature of the decision variables.