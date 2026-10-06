##### Decision Variables

For each supplier $i \in I$ and customer $j \in J$,
$$
x_{ij} \geq 0
$$
where $x_{ij}$ is the quantity shipped from supplier $i$ to customer $j$ (continuous).

##### Sets

- $I$: set of suppliers (stores), from supply_capacity.csv and transportation_costs.csv row labels:
  $$
  I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}\}
  $$
- $J$: set of customers, from customer_demand.csv and transportation_costs.csv column labels:
  $$
  J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}
  $$

##### Parameters

- $d_j$: demand of customer $j$, from customer_demand.csv (table_id: file_0_view_0, column: demand)
- $s_i$: supply capacity of supplier $i$, from supply_capacity.csv (table_id: file_1_view_0, column: supply_capacity)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from transportation_costs.csv (table_id: file_2_view_0, columns: C1–C12, rows: S1–S11)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   $$
2. **Supply capacity** (each supplier ships no more than its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (suppliers): all values in column "Unnamed: 0" of supply_capacity.csv (table_id: file_1_view_0) and row labels of transportation_costs.csv (table_id: file_2_view_0)
- $J$ (customers): all values in column "customer" of customer_demand.csv (table_id: file_0_view_0) and column labels of transportation_costs.csv (table_id: file_2_view_0)
- $d_j$: value in column "demand" for customer $j$ in customer_demand.csv (table_id: file_0_view_0)
- $s_i$: value in column "supply_capacity" for supplier $i$ in supply_capacity.csv (table_id: file_1_view_0)
- $c_{ij}$: value in column $j$ and row $i$ of transportation_costs.csv (table_id: file_2_view_0, columns: C1–C12, rows: S1–S11)

---

**All sets, parameters, and coefficients are to be taken exactly as listed in the referenced CSV files and table_ids.**