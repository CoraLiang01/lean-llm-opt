##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Supermarket demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation (unconditional):**  
   No explicit upper bound on $x_{ij}$ is required unless a capacity is specified. Since only a fixed cost is incurred upon opening, and no capacity is given, $x_{ij}$ are only restricted by demand constraints.
3. **Domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$ (supplier indices from "fixed_cost.csv" and "transportation_costs.csv" Unnamed: 0 column)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$ (supermarket/customer indices from "demand.csv" and "transportation_costs.csv" columns)
- $d_j$: demand for supermarket $j$ from "demand.csv" (column "demand", indexed by "customer")
- $f_i$: fixed cost for supplier $i$ from "fixed_cost.csv" (column "fixed_costs", indexed by "Unnamed: 0")
- $c_{ij}$: transportation cost per unit from supplier $i$ to supermarket $j$ from "transportation_costs.csv" (row "Unnamed: 0" for $i$, column $j$)

**Source-column Data Mapping:**
- "demand.csv": customer $\to$ $j$, demand $\to$ $d_j$
- "fixed_cost.csv": Unnamed: 0 $\to$ $i$, fixed_costs $\to$ $f_i$
- "transportation_costs.csv": Unnamed: 0 $\to$ $i$, $Ck$ columns $\to$ $c_{ij}$ for $j = Ck$