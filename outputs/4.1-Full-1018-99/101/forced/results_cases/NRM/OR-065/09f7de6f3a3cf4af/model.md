#### Index Sets

- $F$: set of warehouses (indexed by $i$), from fixed_cost.csv and transportation_costs.csv ("Unnamed: 0" in both).
- $C$: set of musicians/bands (indexed by $j$), from demand.csv ("customer") and transportation_costs.csv (columns "C1", "C2", ...).

#### Parameters

- $f_i$: fixed cost to activate warehouse $i \in F$ (from fixed_cost.csv, column "fixed_costs").
- $t_{ij}$: transportation cost per unit from warehouse $i \in F$ to customer $j \in C$ (from transportation_costs.csv, entry at row $i$, column $j$).
- $d_j$: demand of customer $j \in C$ (from demand.csv, column "demand").

#### Decision Variables

- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise.
- $x_{ij} \geq 0$: quantity of goods shipped from warehouse $i$ to customer $j$.

#### Objective

Minimize total cost (fixed + transportation):
$$
\min \quad \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij}
$$

#### Constraints

1. **Demand Satisfaction** (each customer's demand must be met):
   $$
   \sum_{i \in F} x_{ij} = d_j, \quad \forall j \in C
   $$

2. **Warehouse Activation** (shipments only from open warehouses):
   $$
   \sum_{j \in C} x_{ij} \leq M_i y_i, \quad \forall i \in F
   $$
   where $M_i$ is a sufficiently large constant (e.g., $M_i = \sum_{j \in C} d_j$).

3. **Variable Domains**:
   $$
   y_i \in \{0,1\}, \quad \forall i \in F
   $$
   $$
   x_{ij} \geq 0, \quad \forall i \in F,\, j \in C
   $$

---

#### Data Mapping

- **Warehouses ($F$):** file_1_view_0 ("Unnamed: 0") and file_2_view_0 ("Unnamed: 0")
- **Musicians/Bands ($C$):** file_0_view_0 ("customer") and file_2_view_0 (columns "C1", "C2", "C3")
- **Fixed Costs ($f_i$):** file_1_view_0 ("fixed_costs")
- **Transportation Costs ($t_{ij}$):** file_2_view_0 (row "Unnamed: 0" for $i$, column "C1"/"C2"/"C3" for $j$)
- **Demands ($d_j$):** file_0_view_0 ("demand")

All data is used as returned by CSVQA, with no additional filtering or aggregation.