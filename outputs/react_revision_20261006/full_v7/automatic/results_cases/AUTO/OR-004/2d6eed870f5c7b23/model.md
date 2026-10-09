##### Mathematical Model

Let $I$ be the set of distribution centers (supply nodes), indexed by $i$, and $J$ the set of customer groups (demand nodes), indexed by $j$.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i \in I$ to customer group $j \in J$.

**Parameters:**
- $d_j$: demand of customer group $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_costs.csv")

**Sets:**
- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Subject to:**

1. **Demand satisfaction:**
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]

2. **Supply capacity:**
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

---

##### Data Mapping

- $I$ (distribution centers): "Unnamed: 0" column in "supply_capacity.csv" and row labels in "transportation_costs.csv"
- $J$ (customer groups): "customer" column in "customer_demand.csv" and column labels in "transportation_costs.csv"
- $d_j$: "demand" column in "customer_demand.csv", indexed by "customer"
- $s_i$: "supply_capacity" column in "supply_capacity.csv", indexed by "Unnamed: 0"
- $c_{ij}$: entry in "transportation_costs.csv" at row $i$ ("Unnamed: 0") and column $j$ (customer group label)

All indices, parameters, and coefficients are to be taken exactly as listed in the respective CSV files.