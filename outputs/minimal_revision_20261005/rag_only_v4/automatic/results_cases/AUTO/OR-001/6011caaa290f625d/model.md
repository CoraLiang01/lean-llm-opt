## Sets
- Let \( S \) be the set of distribution centers (sources), indexed by \( s \), where \( S = \{\text{S1}, \text{S2}, \ldots, \text{S18}\} \) from `supply_capacity.csv` ("file_1_view_0", column "Unnamed: 0").
- Let \( C \) be the set of customer groups, indexed by \( c \), where \( C = \{\text{C1}, \text{C2}, \ldots, \text{C18}\} \) from `customer_demand.csv` ("file_0_view_0", column "customer").

## Parameters
- \( d_c \): Daily demand of customer group \( c \).  
  Data: `customer_demand.csv` ("file_0_view_0", columns "customer", "demand").
- \( u_s \): Daily supply capacity of distribution center \( s \).  
  Data: `supply_capacity.csv` ("file_1_view_0", columns "Unnamed: 0", "supply_capacity").
- \( t_{s,c} \): Transportation cost per unit from distribution center \( s \) to customer group \( c \).  
  Data: `transportation_costs.csv` ("file_2_view_0", row "Unnamed: 0" = \( s \), column \( c \)).

## Decision Variables
- \( x_{s,c} \geq 0 \): Quantity of goods transported from distribution center \( s \) to customer group \( c \).

## Objective
Minimize total transportation cost:
\[
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
\]

## Constraints

1. **Demand satisfaction for each customer group:**
   \[
   \sum_{s \in S} x_{s,c} = d_c \quad \forall c \in C
   \]

2. **Supply capacity for each distribution center:**
   \[
   \sum_{c \in C} x_{s,c} \leq u_s \quad \forall s \in S
   \]

3. **Non-negativity:**
   \[
   x_{s,c} \geq 0 \quad \forall s \in S,\, c \in C
   \]

---

## Data Mapping

- \( S \): All values in `supply_capacity.csv` ("file_1_view_0", column "Unnamed: 0")
- \( C \): All values in `customer_demand.csv` ("file_0_view_0", column "customer")
- \( d_c \): For each \( c \), value in `customer_demand.csv` ("file_0_view_0", columns "customer", "demand")
- \( u_s \): For each \( s \), value in `supply_capacity.csv` ("file_1_view_0", columns "Unnamed: 0", "supply_capacity")
- \( t_{s,c} \): For each \( s \), \( c \), value in `transportation_costs.csv` ("file_2_view_0", row "Unnamed: 0" = \( s \), column \( c \))

---

**All indices, parameters, and coefficients are bound exactly to the data as described above.**