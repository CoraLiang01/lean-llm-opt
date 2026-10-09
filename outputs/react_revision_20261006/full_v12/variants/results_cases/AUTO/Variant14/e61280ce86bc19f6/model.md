##### Decision Variables

$y_i \in \{0,1\}$: 1 if depot $i$ is opened, 0 otherwise, for each candidate depot $i$ (from Center in file_0_view_0).

##### Objective Function

$\min \sum_{i \in I} c_i y_i$

where $I$ is the set of candidate depots (Centers from file_0_view_0), and $c_i$ is the OpeningCost for depot $i$ (OpeningCost from file_0_view_0).

##### Constraints

For each service zone $j$ (Zone in file_1_view_0):

$\sum_{i \in I_j} y_i \geq 1, \quad \forall j \in J$

where $J$ is the set of service zones (Zones from file_1_view_0), and $I_j = \{ i \in I : j \in \text{CoveredDistricts}_i \}$, i.e., the set of depots $i$ whose CoveredDistricts (from file_0_view_0) include zone $j$.

##### Variable Domains

$y_i \in \{0,1\}, \quad \forall i \in I$

---

###### Data Mapping

- Candidate depots $I$: Center column in file_0_view_0 (facility_sites.csv)
- Opening costs $c_i$: OpeningCost column in file_0_view_0 (facility_sites.csv)
- Service zones $J$: Zone column in file_1_view_0 (service_zones.csv)
- Coverage sets $I_j$: For each $j$ in Zone (file_1_view_0), $I_j$ is the set of Centers in file_0_view_0 whose CoveredDistricts (semicolon-separated) include $j$.

All sets and parameters are defined directly from the CSV columns as described.