## Sets
- Let \( S \) be the set of distribution centers, indexed by \( s \), where \( S = \{\text{S1}, \text{S2}, \ldots, \text{S18}\} \) from `file_1_view_0.Unnamed: 0`.
- Let \( C \) be the set of customer groups, indexed by \( c \), where \( C = \{\text{C1}, \text{C2}, \ldots, \text{C18}\} \) from `file_0_view_0.customer`.

## Parameters
- \( d_c \): Daily demand of customer group \( c \), from `file_0_view_0.demand`.
- \( u_s \): Daily supply capacity of distribution center \( s \), from `file_1_view_0.supply_capacity`.
- \( t_{s,c} \): Transportation cost per unit from distribution center \( s \) to customer group \( c \), from `file_2_view_0` with row \( s \) (`Unnamed: 0`) and column \( c \).

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

- \( S \): All values in `file_1_view_0.Unnamed: 0`
- \( C \): All values in `file_0_view_0.customer`
- \( d_c \): `file_0_view_0.demand` where `customer` = \( c \)
- \( u_s \): `file_1_view_0.supply_capacity` where `Unnamed: 0` = \( s \)
- \( t_{s,c} \): `file_2_view_0` entry at row `Unnamed: 0` = \( s \), column \( c \)

---

**This is a linear programming transportation problem with all data and indices mapped directly to the provided tables.**