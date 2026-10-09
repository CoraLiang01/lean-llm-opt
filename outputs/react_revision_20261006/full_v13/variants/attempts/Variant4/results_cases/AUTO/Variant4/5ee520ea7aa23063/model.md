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

- $I$: Set of candidate service centers, from table_id file_0_view_0, column "Center".
- $J$: Set of districts to be covered, from table_id file_1_view_0, column "District".
- $c_i$: Opening cost of center $i$, from file_0_view_0, column "OpeningCost".
- $S_i$: For each $i \in I$, the set of districts covered by center $i$, parsed from file_0_view_0, column "CoveredDistricts" (semicolon-separated list).
- For each $j \in J$, the set $\{i \in I : j \in S_i\}$ is determined by matching $j$ to the entries in $S_i$.

##### Data Mapping

- Centers $I$: file_0_view_0, column "Center"
- Opening costs $c_i$: file_0_view_0, column "OpeningCost"
- Covered districts $S_i$: file_0_view_0, column "CoveredDistricts"
- Districts $J$: file_1_view_0, column "District"