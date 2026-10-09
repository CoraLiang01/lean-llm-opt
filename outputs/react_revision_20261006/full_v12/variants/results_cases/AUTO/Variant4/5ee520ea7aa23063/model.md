##### Decision Variables

$y_i \in \{0,1\}$: $1$ if service center $i$ is opened, $0$ otherwise, for each $i \in I$.

##### Objective Function

$\min \sum_{i \in I} c_i y_i$

##### Constraints

1. Coverage: For every district $j \in J$,
   $$
   \sum_{i \in I: j \in S_i} y_i \geq 1
   $$
   where $S_i$ is the set of districts covered by center $i$.

2. Binary restrictions: $y_i \in \{0,1\}$ for all $i \in I$.

##### Index Sets and Data Mapping

- $I$: set of candidate centers, from service_centers.csv, column "Center", table_id: file_0_view_0.
- $J$: set of districts to cover, from districts.csv, column "District", table_id: file_1_view_0.
- $c_i$: opening cost of center $i$, from service_centers.csv, column "OpeningCost", table_id: file_0_view_0.
- $S_i$: set of districts covered by center $i$, from service_centers.csv, column "CoveredDistricts" (semicolon-separated), table_id: file_0_view_0.

Each constraint for district $j$ sums over all centers $i$ such that $j$ appears in $S_i$. All variables and parameters are mapped directly to the indicated columns and table_ids.