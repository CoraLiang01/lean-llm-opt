##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups (demands):

- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer group $j \in J$.

Parameters:
- $d_j$: demand for customer group $j$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of supplier $i$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ (from file_2_view_0, columns "demand1"–"demand8", rows "supply1"–"supply8")

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
\]
\[
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\]

##### Data Mapping

- $I$ (suppliers): file_1_view_0, column "Supplier", values in source order.
- $J$ (customer groups): file_0_view_0, column "Customers", values in source order.
- $d_j$: file_0_view_0, column "demand", indexed by "Customers".
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "Supplier".
- $c_{ij}$: file_2_view_0, columns "demand1"–"demand8", rows "Unnamed: 0" ("supply1"–"supply8"), with $i$ mapped to supplier and $j$ mapped to demand by source order.

All indices, parameters, and constraints are defined exactly as in the current source data. No data is omitted or aggregated. Variable domains and all bounds are as specified in the user query and source files.