##### Decision Variables

$y_i \in \{0,1\}$: 1 if depot $i \in I$ is opened, 0 otherwise.

##### Objective Function

$\min \sum_{i \in I} c_i y_i$

##### Constraints

1. Coverage: For each service zone $j \in J$,
   $$
   \sum_{i \in I: j \in S_i} y_i \geq 1
   $$
   where $S_i$ is the set of service zones covered by depot $i$.

2. Binary restrictions: $y_i \in \{0,1\}$ for all $i \in I$.

##### Data Mapping

- $I$: Set of candidate depots, from file_0_view_0, column "Center".
- $J$: Set of service zones, from file_1_view_0, column "Zone".
- $c_i$: Opening cost for depot $i$, from file_0_view_0, column "OpeningCost".
- $S_i$: For each depot $i$, the set of service zones it covers, from file_0_view_0, column "CoveredDistricts" (semicolon-separated list).
- Each constraint for zone $j$ includes all $i$ such that $j$ appears in $S_i$.

All sets and parameters are defined exactly by the current CSV data.