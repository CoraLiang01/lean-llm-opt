Let $x_i$ be the quantity of baked good $i$ to fulfill, for each product as identified by "Product Name" in the data.

**Objective:**
\[
\max \sum_{i} r_i x_i
\]
where $r_i$ is the "Revenue" for product $i$.

**Constraints:**

For each product $i$ (using the original "Product Name" as the index):

1. **Inventory constraint:**
\[
x_i \leq \text{Initial Inventory}_i
\]

2. **Demand constraint:**
\[
x_i \leq \text{Demand}_i
\]

3. **Nonnegativity and integrality:**
\[
x_i \in \mathbb{Z}_{\geq 0}
\]

**Explicitly, for each product (in source order):**

For each row in the data, with $i$ as the "Product Name", $r_i$ as "Revenue", $d_i$ as "Demand", and $s_i$ as "Initial Inventory":

- Decision variable: $x_i$ = quantity of $i$ to fulfill

- Constraints:
    - $x_i \leq d_i$
    - $x_i \leq s_i$
    - $x_i \in \mathbb{Z}_{\geq 0}$

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{i} r_i x_i \\
\text{s.t.} \quad & x_i \leq d_i \quad \forall i \\
                  & x_i \leq s_i \quad \forall i \\
                  & x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
\]

where the index $i$ runs over all "Product Name" values in the supplied data, with $r_i$, $d_i$, and $s_i$ as the corresponding "Revenue", "Demand", and "Initial Inventory" values from the same row.