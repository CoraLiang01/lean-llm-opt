##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers), indexed by $i$ (from the supplier_id column in supply_capacity.csv and transportation_costs.csv).  
Let $J$ be the set of customer groups (demands), indexed by $j$ (from the customer_id column in customer_demand.csv and transportation_costs.csv).

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer group $j \in J$.

Parameters:
- $d_j$: demand of customer group $j$ (from customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ (from transportation_costs.csv)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (suppliers): All unique values in supply_capacity.csv supplier_id and transportation_costs.csv supplier_id, in source order:  
  file_1_view_0.supplier_id, file_2_view_0.supplier_id

- $J$ (customer groups): All unique values in customer_demand.csv customer_id and transportation_costs.csv column suffixes, in source order:  
  file_0_view_0.customer_id, file_2_view_0 columns transportation_cost_to_{customer_id}

- $d_j$: file_0_view_0.demand, indexed by file_0_view_0.customer_id

- $s_i$: file_1_view_0.supply_capacity, indexed by file_1_view_0.supplier_id

- $c_{ij}$: file_2_view_0.transportation_cost_to_{customer_id}, indexed by file_2_view_0.supplier_id (rows) and customer_id (columns), using the relationships mapping in the Observation

- $x_{ij}$: Decision variable for each $(i,j)$ pair as above

All index sets, parameters, and constraints are defined exactly as in the current CSV data and relationships. No data is omitted or aggregated. Variable domains and all bounds are as specified in the user query and source data.