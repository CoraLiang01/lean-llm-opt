##### Decision Variables

$y_i \in \{0,1\}$: 1 if service center $i \in I$ is opened, 0 otherwise.

##### Objective Function

$\min \sum_{i \in I} c_i y_i$

##### Constraints

1. Coverage: For every district $j \in J$,
   $$
   \sum_{i \in I: j \in S_i} y_i \geq 1, \quad \forall j \in J
   $$
   where $S_i$ is the set of districts covered by center $i$.

2. Binary restrictions: $y_i \in \{0,1\}$ for all $i \in I$.

##### Index Sets and Data Mapping

- $I$: Set of candidate service centers, from column "Center" in table_id file_0_view_0 (service_centers.csv).
- $J$: Set of demand districts, from column "District" in table_id file_1_view_0 (districts.csv).
- $c_i$: Opening cost for center $i$, from column "OpeningCost" in table_id file_0_view_0 (service_centers.csv).
- $S_i$: For each $i \in I$, the set of districts covered by center $i$, parsed from column "CoveredDistricts" in table_id file_0_view_0 (service_centers.csv), with districts separated by ";".
- Each constraint for district $j$ includes all $i$ such that $j \in S_i$.

All parameters and sets are defined directly from the CSV data as described above.