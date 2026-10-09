#### Mathematical Model

Let $S$ be the set of warehouses (supplier_id from file_1), $D$ the set of stores (customer_id from file_0).

Let $x_{sd} \geq 0$ be the quantity shipped from warehouse $s \in S$ to store $d \in D$.

**Objective:**
\[
\min \sum_{s \in S} \sum_{d \in D} c_{sd} \, x_{sd}
\]
where $c_{sd}$ is the transportation cost per unit from warehouse $s$ to store $d$.

**Constraints:**
1. **Demand satisfaction:**  
   For each store $d \in D$,
   \[
   \sum_{s \in S} x_{sd} \geq \text{demand}_d
   \]
   where $\text{demand}_d$ is the demand_units for store $d$.

2. **Supply capacity:**  
   For each warehouse $s \in S$,
   \[
   \sum_{d \in D} x_{sd} \leq \text{supply\_capacity}_s
   \]
   where $\text{supply\_capacity}_s$ is the supply_capacity_units for warehouse $s$.

3. **Non-negativity:**  
   \[
   x_{sd} \geq 0 \quad \forall s \in S,\, d \in D
   \]

#### Data Mapping

- $S$ (warehouses): supplier_id from file_1_view_0 (supply_capacity.csv)
- $D$ (stores): customer_id from file_0_view_0 (customer_demand.csv)
- $\text{demand}_d$: demand_units from file_0_view_0, column "demand_units", indexed by customer_id
- $\text{supply\_capacity}_s$: supply_capacity_units from file_1_view_0, column "supply_capacity_units", indexed by supplier_id
- $c_{sd}$: transportation_cost_to_D* from file_2_view_0 (transportation_costs.csv), where each row is supplier_id $s$ and each column "transportation_cost_to_Dk" corresponds to customer_id $Dk$.

All index sets, parameters, and coefficients are defined exactly as in the current CSV data. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.