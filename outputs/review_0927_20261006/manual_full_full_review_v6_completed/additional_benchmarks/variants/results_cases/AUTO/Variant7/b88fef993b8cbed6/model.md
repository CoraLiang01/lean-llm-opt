##### Sets

- Sources: $S = \{S1, S2, S3\}$
- Hubs: $H = \{H1, H2\}$
- Customers: $C = \{C1, C2, C3, C4\}$
- Nodes: $N = S \cup H \cup C$
- Arcs: $(i,j)$ as listed below

##### Parameters

- Source supplies:
  - $s_{S1} = 120$
  - $s_{S2} = 100$
  - $s_{S3} = 90$
- Customer demands:
  - $d_{C1} = 70$
  - $d_{C2} = 80$
  - $d_{C3} = 60$
  - $d_{C4} = 90$
- Hub throughput capacities:
  - $u_{H1} = 170$
  - $u_{H2} = 160$
- Arc costs $c_{ij}$ (for each directed arc $(i,j)$):

| From | To | $c_{ij}$ |
|------|----|----------|
| S1   | H1 | 2        |
| S1   | H2 | 6        |
| S2   | H1 | 4        |
| S2   | H2 | 3        |
| S3   | H1 | 7        |
| S3   | H2 | 2        |
| H1   | C1 | 3        |
| H1   | C2 | 4        |
| H1   | C3 | 7        |
| H1   | C4 | 8        |
| H2   | C1 | 8        |
| H2   | C2 | 6        |
| H2   | C3 | 3        |
| H2   | C4 | 4        |

##### Decision Variables

- $f_{ij} \geq 0$ : shipment flow on arc $(i,j)$, for all arcs listed above.

##### Objective

Minimize total transportation cost:
$$
\min \sum_{(i,j)} c_{ij} f_{ij}
$$

##### Constraints

1. **Source supply upper bounds** (for each $s \in S$):
   $$
   \sum_{h \in H} f_{s h} \leq s_s
   $$
   - $f_{S1,H1} + f_{S1,H2} \leq 120$
   - $f_{S2,H1} + f_{S2,H2} \leq 100$
   - $f_{S3,H1} + f_{S3,H2} \leq 90$

2. **Customer demand constraints** (for each $c \in C$):
   $$
   \sum_{h \in H} f_{h c} \geq d_c
   $$
   - $f_{H1,C1} + f_{H2,C1} \geq 70$
   - $f_{H1,C2} + f_{H2,C2} \geq 80$
   - $f_{H1,C3} + f_{H2,C3} \geq 60$
   - $f_{H1,C4} + f_{H2,C4} \geq 90$

3. **Flow balance at each hub $h \in H$**:
   $$
   \sum_{s \in S} f_{s h} = \sum_{c \in C} f_{h c}
   $$
   - $f_{S1,H1} + f_{S2,H1} + f_{S3,H1} = f_{H1,C1} + f_{H1,C2} + f_{H1,C3} + f_{H1,C4}$
   - $f_{S1,H2} + f_{S2,H2} + f_{S3,H2} = f_{H2,C1} + f_{H2,C2} + f_{H2,C3} + f_{H2,C4}$

4. **Hub throughput capacity** (for each $h \in H$):
   $$
   \sum_{s \in S} f_{s h} \leq u_h
   $$
   - $f_{S1,H1} + f_{S2,H1} + f_{S3,H1} \leq 170$
   - $f_{S1,H2} + f_{S2,H2} + f_{S3,H2} \leq 160$

5. **Nonnegativity**:
   $$
   f_{ij} \geq 0 \quad \text{for all arcs } (i,j)
   $$

##### Arc List (for variable and cost reference)

- $(S1,H1), (S1,H2), (S2,H1), (S2,H2), (S3,H1), (S3,H2)$
- $(H1,C1), (H1,C2), (H1,C3), (H1,C4), (H2,C1), (H2,C2), (H2,C3), (H2,C4)$

##### Complete Model

Minimize
$$
2 f_{S1,H1} + 6 f_{S1,H2} + 4 f_{S2,H1} + 3 f_{S2,H2} + 7 f_{S3,H1} + 2 f_{S3,H2} \\
+ 3 f_{H1,C1} + 4 f_{H1,C2} + 7 f_{H1,C3} + 8 f_{H1,C4} \\
+ 8 f_{H2,C1} + 6 f_{H2,C2} + 3 f_{H2,C3} + 4 f_{H2,C4}
$$

Subject to:
\[
\begin{align*}
& f_{S1,H1} + f_{S1,H2} \leq 120 \\
& f_{S2,H1} + f_{S2,H2} \leq 100 \\
& f_{S3,H1} + f_{S3,H2} \leq 90 \\
& f_{H1,C1} + f_{H2,C1} \geq 70 \\
& f_{H1,C2} + f_{H2,C2} \geq 80 \\
& f_{H1,C3} + f_{H2,C3} \geq 60 \\
& f_{H1,C4} + f_{H2,C4} \geq 90 \\
& f_{S1,H1} + f_{S2,H1} + f_{S3,H1} = f_{H1,C1} + f_{H1,C2} + f_{H1,C3} + f_{H1,C4} \\
& f_{S1,H2} + f_{S2,H2} + f_{S3,H2} = f_{H2,C1} + f_{H2,C2} + f_{H2,C3} + f_{H2,C4} \\
& f_{S1,H1} + f_{S2,H1} + f_{S3,H1} \leq 170 \\
& f_{S1,H2} + f_{S2,H2} + f_{S3,H2} \leq 160 \\
& f_{ij} \geq 0 \quad \text{for all arcs } (i,j)
\end{align*}
\]

###### Retrieved Information

{
  "sources": ["S1", "S2", "S3"],
  "hubs": ["H1", "H2"],
  "customers": ["C1", "C2", "C3", "C4"],
  "source_supply": {"S1": 120, "S2": 100, "S3": 90},
  "customer_demand": {"C1": 70, "C2": 80, "C3": 60, "C4": 90},
  "hub_capacity": {"H1": 170, "H2": 160},
  "arc_costs": [
    {"From": "S1", "To": "H1", "Cost": 2},
    {"From": "S1", "To": "H2", "Cost": 6},
    {"From": "S2", "To": "H1", "Cost": 4},
    {"From": "S2", "To": "H2", "Cost": 3},
    {"From": "S3", "To": "H1", "Cost": 7},
    {"From": "S3", "To": "H2", "Cost": 2},
    {"From": "H1", "To": "C1", "Cost": 3},
    {"From": "H1", "To": "C2", "Cost": 4},
    {"From": "H1", "To": "C3", "Cost": 7},
    {"From": "H1", "To": "C4", "Cost": 8},
    {"From": "H2", "To": "C1", "Cost": 8},
    {"From": "H2", "To": "C2", "Cost": 6},
    {"From": "H2", "To": "C3", "Cost": 3},
    {"From": "H2", "To": "C4", "Cost": 4}
  ]
}