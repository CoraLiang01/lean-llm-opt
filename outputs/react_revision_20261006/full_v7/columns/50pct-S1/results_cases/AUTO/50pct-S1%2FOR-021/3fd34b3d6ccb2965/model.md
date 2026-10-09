##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of beverages shipped from production plant $i$ to retail outlet $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets)

##### Parameters

- $d_j$: demand at outlet $j$ (from customer_demand.csv)
    - $d_{\text{C1}} = 94$
    - $d_{\text{C2}} = 39$
    - $d_{\text{C3}} = 65$
    - $d_{\text{C4}} = 435$
- $s_i$: supply capacity at plant $i$ (from supply_capacity.csv)
    - $s_{\text{S1}} = 2531$
    - $s_{\text{S2}} = 20$
    - $s_{\text{S3}} = 210$
    - $s_{\text{S4}} = 241$
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv)

|         | C1                | C2                | C3                | C4                |
|---------|-------------------|-------------------|-------------------|-------------------|
| S1      | 543.756480860856  | 23.685276141764653| 23.676386730773032| 447.75143678673766|
| S2      | 883.9151090405642 | 0.04977684765576961| 0.0350986687216299| 44.45588531711622 |
| S3      | 537.3456896658107 | 23.769274659075112| 498.95659249465467| 440.60737890439776|
| S4      | 1791.493192397229 | 68.21633865655126 | 1432.4837339656747| 1527.7635425462734|

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. Demand satisfaction at each outlet:
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. Supply capacity at each plant:
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. Non-negativity:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (plants): supplier_id in supply_capacity.csv and transportation_costs.csv
- $J$ (outlets): customer_id in customer_demand.csv and columns in transportation_costs.csv (suffix after "transportation_cost_to_")
- $d_j$: demand column in customer_demand.csv, indexed by customer_id
- $s_i$: supply_capacity column in supply_capacity.csv, indexed by supplier_id
- $c_{ij}$: transportation_costs.csv, row supplier_id $i$, column "transportation_cost_to_$j$"

All indices, parameters, and coefficients are mapped directly from the provided CSV files as described above.