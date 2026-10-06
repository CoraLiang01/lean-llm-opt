Sets:
- S: Set of production plants, indexed by s. S = {S1, S2, S3, S4} (from file_1_view_0, column "Unnamed: 0")
- C: Set of retail outlets, indexed by c. C = {C1, C2, C3, C4} (from file_0_view_0, column "customer")

Parameters:
- demand_c: Daily demand at outlet c ∈ C. (from file_0_view_0, column "demand")
- supply_capacity_s: Daily production capacity at plant s ∈ S. (from file_1_view_0, column "supply_capacity")
- cost_sc: Transportation cost per unit from plant s to outlet c. (from file_2_view_0, columns "C1", "C2", "C3", "C4", rows indexed by "Unnamed: 0")

Decision Variables:
- x_sc ≥ 0: Quantity of beverages shipped from plant s ∈ S to outlet c ∈ C

Mathematical Model:

Minimize total transportation cost:
\[
\min \sum_{s \in S} \sum_{c \in C} \text{cost}_{sc} \cdot x_{sc}
\]

Subject to:

1. Demand satisfaction at each outlet:
\[
\sum_{s \in S} x_{sc} = \text{demand}_c \quad \forall c \in C
\]

2. Plant capacity limits:
\[
\sum_{c \in C} x_{sc} \leq \text{supply\_capacity}_s \quad \forall s \in S
\]

3. Nonnegativity:
\[
x_{sc} \geq 0 \quad \forall s \in S,\, c \in C
\]

Data Mapping:

- S = {S1, S2, S3, S4} from file_1_view_0["Unnamed: 0"]
- C = {C1, C2, C3, C4} from file_0_view_0["customer"]
- demand_c: file_0_view_0["demand"], indexed by file_0_view_0["customer"]
- supply_capacity_s: file_1_view_0["supply_capacity"], indexed by file_1_view_0["Unnamed: 0"]
- cost_sc: file_2_view_0, rows indexed by "Unnamed: 0" (plants S), columns "C1", "C2", "C3", "C4" (customers C)

Variable domains:
- x_sc ≥ 0 for all s ∈ S, c ∈ C

Objective sense:
- Minimize total transportation cost

All sets, parameters, and constraints are bound directly to the supplied data.