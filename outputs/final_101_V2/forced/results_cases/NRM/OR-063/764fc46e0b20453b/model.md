#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of warehouses (from fixed_cost.csv, column "Unnamed: 0")
- $C$: set of musicians/bands (from demand.csv, column "customer")

**Parameters:**
- $f_i$: fixed cost of opening warehouse $i \in S$ (from fixed_cost.csv, column "fixed_costs")
- $d_j$: demand of musician/band $j \in C$ (from demand.csv, column "demand")
- $t_{ij}$: transportation cost per unit from warehouse $i \in S$ to musician/band $j \in C$ (from transportation_costs.csv, columns "Unnamed: 0" for $i$, columns $C$ for $j$)

**Decision Variables:**
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise, $\forall i \in S$
- $x_{ij} \geq 0$: quantity of goods shipped from warehouse $i$ to musician/band $j$, $\forall i \in S, j \in C$

**Objective:**
\[
\min \left( \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} t_{ij} x_{ij} \right)
\]

**Constraints:**

1. **Demand Satisfaction:**
   \[
   \sum_{i \in S} x_{ij} = d_j, \quad \forall j \in C
   \]

2. **Warehouse Activation:**
   \[
   x_{ij} \leq d_j y_i, \quad \forall i \in S, \forall j \in C
   \]

3. **Variable Domains:**
   \[
   y_i \in \{0,1\}, \quad \forall i \in S
   \]
   \[
   x_{ij} \geq 0, \quad \forall i \in S, \forall j \in C
   \]

---

#### Data Mapping

- **Warehouses ($S$) and Fixed Costs ($f_i$):**
  - Source: fixed_cost.csv
  - Table ID: file_1_view_0
  - Columns: "Unnamed: 0" (warehouse identifier), "fixed_costs" (fixed cost)

- **Musicians/Bands ($C$) and Demands ($d_j$):**
  - Source: demand.csv
  - Table ID: file_0_view_0
  - Columns: "customer" (musician/band identifier), "demand" (demand)

- **Transportation Costs ($t_{ij}$):**
  - Source: transportation_costs.csv
  - Table ID: file_2_view_0
  - Row: "Unnamed: 0" (warehouse $i$)
  - Columns: $C$ (musician/band $j$)

---

All sets, parameters, and relationships are defined symbolically and mapped to their exact source columns and table IDs. No literal data values or record counts are included.