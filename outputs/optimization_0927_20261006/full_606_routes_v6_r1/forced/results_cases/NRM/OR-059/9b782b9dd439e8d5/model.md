#### Index Sets

- $I$: set of suppliers (from file_1_view_0, column "Unnamed: 0" in fixed_cost.csv)
- $J$: set of dealerships (from file_0_view_0, column "customer" in demand.csv)

#### Parameters

- $f_i$: fixed cost of opening supplier $i \in I$ (from file_1_view_0, column "fixed_costs" in fixed_cost.csv)
- $c_{ij}$: transportation cost per vehicle from supplier $i \in I$ to dealership $j \in J$ (from file_2_view_0, columns "Unnamed: 0" for suppliers and "C1", ..., "C9" for dealerships in transportation_costs.csv)
- $d_j$: demand of dealership $j \in J$ (from file_0_view_0, column "demand" in demand.csv)

#### Decision Variables

- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: number of vehicles supplied from supplier $i$ to dealership $j$

#### Objective

$$
\min \left( \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \right)
$$

#### Constraints

1. **Demand Satisfaction (for each dealership):**
   $$
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   $$

2. **Supplier Activation (flow only if open):**
   $$
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   $$
   where $M_i$ is a sufficiently large constant (e.g., $M_i = \sum_{j \in J} d_j$).

3. **Variable Domains:**
   $$
   y_i \in \{0,1\}, \quad \forall i \in I
   $$
   $$
   x_{ij} \geq 0, \quad \forall i \in I, \forall j \in J
   $$

---

#### Data Mapping

- **file_1_view_0 (fixed_cost.csv):**
  - Index set $I$ from column "Unnamed: 0"
  - Parameter $f_i$ from column "fixed_costs"
- **file_0_view_0 (demand.csv):**
  - Index set $J$ from column "customer"
  - Parameter $d_j$ from column "demand"
- **file_2_view_0 (transportation_costs.csv):**
  - Parameter $c_{ij}$ from columns "Unnamed: 0" (supplier IDs) and "C1", ..., "C9" (dealership IDs), with $i$ from "Unnamed: 0" and $j$ from column headers

All data is used as returned by CSVQA, with no additional filtering or transformation.