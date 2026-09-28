#### Abstract Facility Location Model

**Index Sets**
- $I$: set of potential factory sites (from facility_costs.csv, column Facility)
- $J$: set of distribution centers (from demand_requirements.csv, column Destination)

**Parameters**
- $f_i$: fixed cost to construct factory $i \in I$ (facility_costs.csv, column FixedCost)
- $c_{ij}$: variable shipping cost per unit from factory $i \in I$ to distribution center $j \in J$ (shipping_costs.csv, entry at row Origin $i$, column $j$)
- $d_j$: demand at distribution center $j \in J$ (demand_requirements.csv, column Demand)
- $K_i$: capacity of factory $i \in I$ (facility_costs.csv, column Capacity)

**Decision Variables**
- $y_i \in \{0,1\}$: 1 if factory $i$ is constructed, 0 otherwise
- $x_{ij} \geq 0$: quantity shipped from factory $i$ to distribution center $j$

**Objective**
\[
\min \quad \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Constraints**
1. **Demand Satisfaction:**  
   $\displaystyle \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J$

2. **Factory Capacity (if constructed):**  
   $\displaystyle \sum_{j \in J} x_{ij} \leq K_i y_i \quad \forall i \in I$

3. **Variable Domains:**  
   $y_i \in \{0,1\} \quad \forall i \in I$  
   $x_{ij} \geq 0 \quad \forall i \in I,\, j \in J$

---

#### Data Mapping

- **Index Set $I$ (factories):** facility_costs.csv, column Facility
- **Index Set $J$ (distribution centers):** demand_requirements.csv, column Destination
- **Parameter $f_i$ (fixed cost):** facility_costs.csv, column FixedCost
- **Parameter $K_i$ (capacity):** facility_costs.csv, column Capacity
- **Parameter $d_j$ (demand):** demand_requirements.csv, column Demand
- **Parameter $c_{ij}$ (shipping cost):** shipping_costs.csv, row Origin $i$, column $j$ (where $i$ from Facility, $j$ from Destination)

No literal data values or record counts are included; all identifiers and column names are preserved as in the source files.