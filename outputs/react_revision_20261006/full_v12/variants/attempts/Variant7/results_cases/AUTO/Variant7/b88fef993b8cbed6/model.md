#### Mathematical Model

Let $N$ be the set of all nodes (sources, hubs, customers), $A$ the set of directed arcs $(i,j)$ with per-unit cost $c_{ij}$, and $f_{ij}\geq0$ the flow on arc $(i,j)\in A$.

**Objective:**
\[
\min \sum_{(i,j)\in A} c_{ij} f_{ij}
\]

**Constraints:**

1. **Source supply upper bounds** (for each source $s$):
   \[
   \sum_{j: (s,j)\in A} f_{sj} \leq \text{Amount}_s
   \]
   where $s$ is any node with $\text{NodeType} = \text{SourceSupply}$ in file_0_view_0, and $\text{Amount}_s$ is its supply.

2. **Customer demand constraints** (for each customer $c$):
   \[
   \sum_{i: (i,c)\in A} f_{ic} \geq \text{Amount}_c
   \]
   where $c$ is any node with $\text{NodeType} = \text{CustomerDemand}$ in file_0_view_0, and $\text{Amount}_c$ is its demand.

3. **Flow balance at each hub $h$**:
   \[
   \sum_{i: (i,h)\in A} f_{ih} = \sum_{j: (h,j)\in A} f_{hj}
   \]
   for all $h$ in the set of hubs (from file_1_view_0, column "Hub").

4. **Hub throughput capacity** (for each hub $h$):
   \[
   \sum_{i: (i,h)\in A} f_{ih} \leq \text{ThroughputCapacity}_h
   \]
   where $\text{ThroughputCapacity}_h$ is from file_1_view_0.

5. **Nonnegativity:**
   \[
   f_{ij} \geq 0 \quad \forall (i,j)\in A
   \]

#### Data Mapping

- **Sources and Customers:** file_0_view_0, columns "Node", "NodeType", "Amount"
    - Sources: rows with NodeType = "SourceSupply"
    - Customers: rows with NodeType = "CustomerDemand"
- **Hubs:** file_1_view_0, column "Hub"
- **Hub capacities:** file_1_view_0, column "ThroughputCapacity"
- **Arcs and costs:** file_2_view_0, columns "From", "To", "Cost"
    - $A = \{(i,j):$ each row in file_2_view_0 with $i=$"From", $j=$"To"$\}$
    - $c_{ij}$ from "Cost" column

- **Variables:** $f_{ij}$ for each $(i,j)\in A$ (from file_2_view_0)

All sets, parameters, and constraints are defined directly from the above tables and columns, preserving all identifiers and coefficients.