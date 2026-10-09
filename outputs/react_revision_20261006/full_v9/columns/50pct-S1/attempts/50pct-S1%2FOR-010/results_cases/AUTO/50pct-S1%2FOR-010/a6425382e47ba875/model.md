Mathematical Optimization Model

Index Sets:
- $S$: Set of sections, indexed by $s$ (SectionID from file_0_view_0)
- $P$: Set of products, indexed by $p$ (ProductName from file_1_view_0)

Parameters:
- $C_s$: Capacity of section $s$ (Capacity from file_0_view_0, indexed by SectionID)
- $v_p$: Value (price) of product $p$ (Value from file_1_view_0, indexed by ProductName)
- $w_p$: Space requirement (Weight) of product $p$ (Weight from file_1_view_0, indexed by ProductName)

Decision Variables:
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ to stock in section $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{s,p}
\]

Subject to:
\[
\sum_{p \in P} w_p \cdot x_{s,p} \leq C_s \qquad \forall s \in S
\]
\[
x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping:
- $S$: SectionID from file_0_view_0 (capacity.csv)
- $P$: ProductName from file_1_view_0 (products.csv)
- $C_s$: file_0_view_0, column Capacity, key SectionID
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName
- $x_{s,p}$: Decision variable for each $(s,p)$ pair

All parameters and index sets are defined directly from the supplied CSV data. Each section's total product space used cannot exceed its listed capacity. All variables are nonnegative integers. The objective is to maximize total revenue.