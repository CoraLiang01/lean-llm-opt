#### Mathematical Model

Let $I$ be the set of Walmart stores (suppliers) and $J$ the set of customer groups, as defined by the current data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from store $i \in I$ to customer group $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer group $j$ (from file_0_view_0, column "demand").
- $s_i$: supply capacity of store $i$ (from file_1_view_0, column "supply_capacity").
- $c_{ij}$: transportation cost per unit from store $i$ to customer group $j$ (from file_2_view_0, columns $J$, rows $I$).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$ (stores): All values in file_1_view_0, column "Unnamed: 0"
- $J$ (customer groups): All values in file_0_view_0, column "customer"
- $d_j$: file_0_view_0, column "demand", indexed by "customer"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "Unnamed: 0"
- $c_{ij}$: file_2_view_0, value at row "Unnamed: 0" = $i$, column $j$ (where $j$ matches "customer" in file_0_view_0)

All sets and parameters are defined by the full, current contents of the referenced files. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.