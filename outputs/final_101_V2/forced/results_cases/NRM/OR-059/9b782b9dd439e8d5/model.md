### Abstract Mathematical Model

#### Index Sets
- $I$: set of suppliers (from `fixed_cost.csv`, column `Unnamed: 0`)
- $J$: set of dealerships (from `demand.csv`, column `customer`)

#### Parameters
- $f_i$: fixed cost to open supplier $i \in I$ (from `fixed_cost.csv`, column `fixed_costs`)
- $c_{ij}$: transportation cost per vehicle from supplier $i \in I$ to dealership $j \in J$ (from `transportation_costs.csv`, columns `Unnamed: 0` for supplier, columns $J$ for dealership)
- $d_j$: demand (number of vehicles) required by dealership $j \in J$ (from `demand.csv`, column `demand`)

#### Decision Variables
- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: number of vehicles supplied from supplier $i$ to dealership $j$

#### Objective
Minimize total cost (fixed + transportation):
$$
\min \quad \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

#### Constraints

1. **Demand Satisfaction (for each dealership):**
   $$
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   $$

2. **Supplier Activation (vehicles can only be supplied if supplier is open):**
   $$
   x_{ij} \leq d_j y_i \quad \forall i \in I, \forall j \in J
   $$

3. **Variable Domains:**
   $$
   y_i \in \{0,1\} \quad \forall i \in I
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in I, \forall j \in J
   $$

---

### Data Mapping

- **Index Set $I$ (Suppliers):** `fixed_cost.csv`, column `Unnamed: 0`
- **Index Set $J$ (Dealerships):** `demand.csv`, column `customer`
- **Parameter $f_i$ (Supplier Fixed Cost):** `fixed_cost.csv`, column `fixed_costs`
- **Parameter $c_{ij}$ (Transportation Cost):** `transportation_costs.csv`, row `Unnamed: 0` (supplier), columns $J$ (dealerships)
- **Parameter $d_j$ (Dealership Demand):** `demand.csv`, column `demand`