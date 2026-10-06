##### Decision Variables

$y_i \in \{0,1\}$: 1 if service center $i \in I$ is opened, 0 otherwise.

##### Parameters

- $I$: set of candidate service centers (from service_centers.csv, column Center, table_id: file_0_view_0)
- $J$: set of demand districts (from districts.csv, column District, table_id: file_1_view_0)
- $c_i$: opening cost of center $i$ (from service_centers.csv, column OpeningCost, table_id: file_0_view_0)
- $C_{ij}$: coverage indicator, 1 if center $i$ covers district $j$, 0 otherwise (from service_centers.csv, column CoveredDistricts, table_id: file_0_view_0, parsed as set membership)

##### Objective Function

\[
\min \sum_{i \in I} c_i y_i
\]

##### Constraints

1. **Coverage:** Every district must be covered by at least one opened center:
   \[
   \sum_{i \in I} C_{ij} y_i \geq 1, \quad \forall j \in J
   \]

2. **Binary restrictions:**
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$: All values in service_centers.csv, column Center (table_id: file_0_view_0)
- $J$: All values in districts.csv, column District (table_id: file_1_view_0)
- $c_i$: service_centers.csv, column OpeningCost, indexed by Center (table_id: file_0_view_0)
- $C_{ij}$: 1 if district $j$ is listed in CoveredDistricts for center $i$ in service_centers.csv (table_id: file_0_view_0, columns Center and CoveredDistricts); 0 otherwise

All sets and parameters are defined exactly as in the source data.