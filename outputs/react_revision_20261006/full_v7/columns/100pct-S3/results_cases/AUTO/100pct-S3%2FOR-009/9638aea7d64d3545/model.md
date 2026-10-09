#### Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), with each area identified by its ProductName from products.csv.
- $x_i$ = scale of development per day in area $i$ (decision variable, nonnegative integer).
- $v_i$ = development benefit per unit in area $i$ (parameter, from Value column in products.csv).
- $w_i$ = resource requirement per unit in area $i$ (parameter, from Weight column in products.csv).
- $C$ = overall development capacity (parameter, from Capacity column in capacity.csv).

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

#### Data Mapping

- $I$: All ProductName values in /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv, column ProductName, all 20 rows.
- $v_i$: /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv, column Value, key ProductName.
- $w_i$: /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv, column Weight, key ProductName.
- $C$: /Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv, column Capacity, row 0.

Variables:
- $x_i$: scale of development per day in area $i$, nonnegative integer, for all $i \in I$.