#### Index Sets

- $I$: set of suppliers (from "fixed_cost.csv" and "transportation_costs.csv" row identifiers)
- $J$: set of supermarkets/customers (from "demand.csv" and "transportation_costs.csv" column identifiers)

#### Parameters

- $f_i$: fixed cost of opening supplier $i \in I$ ("fixed_cost.csv", column "fixed_costs", indexed by "Unnamed: 0")
- $c_{ij}$: transportation cost per unit from supplier $i \in I$ to supermarket $j \in J$ ("transportation_costs.csv", columns $J$, indexed by "Unnamed: 0")
- $d_j$: demand of supermarket $j \in J$ ("demand.csv", column "demand", indexed by "customer")

#### Decision Variables

- $y_i \in \{0,1\}$: 1 if supplier $i$ is open, 0 otherwise
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to supermarket $j$

#### Objective

$$
\min \quad \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

#### Constraints

1. **Demand Satisfaction** (each supermarket's demand must be met):

$$
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
$$

2. **Supplier Activation** (can only supply if open):

$$
x_{ij} \leq d_j y_i \quad \forall i \in I, \forall j \in J
$$

3. **Variable Domains**:

$$
y_i \in \{0,1\} \quad \forall i \in I
$$

$$
x_{ij} \geq 0 \quad \forall i \in I, \forall j \in J
$$

---

#### Data Mapping

- "fixed_cost.csv": supplier set $I$ from column "Unnamed: 0"; parameter $f_i$ from column "fixed_costs"
- "demand.csv": supermarket set $J$ from column "customer"; parameter $d_j$ from column "demand"
- "transportation_costs.csv": supplier set $I$ from column "Unnamed: 0"; supermarket set $J$ from columns "C1", ..., "C25"; parameter $c_{ij}$ from corresponding cell at row $i$, column $j$