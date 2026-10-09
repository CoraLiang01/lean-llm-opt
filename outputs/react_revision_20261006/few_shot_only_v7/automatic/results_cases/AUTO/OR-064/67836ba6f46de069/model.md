##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Supermarket demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation (unconditional bounds):**  
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Sets and Parameters

- $I = \{$S1, S2, ..., S24$\}$: set of suppliers (from "fixed_cost.csv" and "transportation_costs.csv" row labels)
- $J = \{$C1, C2, ..., C25$\}$: set of supermarkets (from "demand.csv" and "transportation_costs.csv" column labels)
- $d_j$: demand of supermarket $j$ (from "demand.csv")
- $f_i$: fixed cost for opening supplier $i$ (from "fixed_cost.csv")
- $c_{ij}$: transportation cost per unit from supplier $i$ to supermarket $j$ (from "transportation_costs.csv")

##### Data Mapping

- **Supplier set $I$:** All "Unnamed: 0" entries in "fixed_cost.csv" and "transportation_costs.csv" rows.
- **Supermarket set $J$:** All "customer" entries in "demand.csv" and columns (except "Unnamed: 0") in "transportation_costs.csv".
- **Demands $d_j$:** "demand" column in "demand.csv", indexed by "customer".
- **Fixed costs $f_i$:** "fixed_costs" column in "fixed_cost.csv", indexed by "Unnamed: 0".
- **Transportation costs $c_{ij}$:** Each entry in "transportation_costs.csv" at row $i$ ("Unnamed: 0") and column $j$ ($C1$ to $C25$).

All variables, sets, and parameters are indexed exactly as in the source CSVs. No capacity or activation-conditioned bounds are imposed beyond nonnegativity and binary activation.