##### Sets

- Sources: $S = \{S1, S2, S3\}$
- Hubs: $H = \{H1, H2\}$
- Customers: $C = \{C1, C2, C3, C4\}$
- Nodes: $N = S \cup H \cup C$
- Arcs: $(i,j)$ as listed in arc_costs.csv

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
- Arc costs $c_{ij}$ (from arc_costs.csv):

| From | To | Cost |
|------|----|------|
| S1   | H1 | 2    |
| S1   | H2 | 6    |
| S2   | H1 | 4    |
| S2   | H2 | 3    |
| S3   | H1 | 7    |
| S3   | H2 | 2    |
| H1   | C1 | 3    |
| H1   | C2 | 4    |
| H1   | C3 | 7    |
| H1   | C4 | 8    |
| H2   | C1 | 8    |
| H2   | C2 | 6    |
| H2   | C3 | 3    |
| H2   | C4 | 4    |

##### Decision Variables

- $f_{ij} \geq 0$: shipment flow on arc $(i,j)$, for each arc listed above (continuous).

##### Objective

Minimize total transportation cost:
$$
\min \sum_{(i,j)} c_{ij} f_{ij}
$$
where the sum is over all arcs listed in arc_costs.csv.

##### Constraints

1. **Source supply upper bounds** (for each $s \in S$):
   $$
   \sum_{h \in H} f_{s h} \leq s_s
   $$
   - For $S1$: $f_{S1,H1} + f_{S1,H2} \leq 120$
   - For $S2$: $f_{S2,H1} + f_{S2,H2} \leq 100$
   - For $S3$: $f_{S3,H1} + f_{S3,H2} \leq 90$

2. **Customer demand constraints** (for each $c \in C$):
   $$
   \sum_{h \in H} f_{h c} \geq d_c
   $$
   - For $C1$: $f_{H1,C1} + f_{H2,C1} \geq 70$
   - For $C2$: $f_{H1,C2} + f_{H2,C2} \geq 80$
   - For $C3$: $f_{H1,C3} + f_{H2,C3} \geq 60$
   - For $C4$: $f_{H1,C4} + f_{H2,C4} \geq 90$

3. **Flow balance at each hub** (for each $h \in H$):
   $$
   \sum_{s \in S} f_{s h} = \sum_{c \in C} f_{h c}
   $$
   - For $H1$: $f_{S1,H1} + f_{S2,H1} + f_{S3,H1} = f_{H1,C1} + f_{H1,C2} + f_{H1,C3} + f_{H1,C4}$
   - For $H2$: $f_{S1,H2} + f_{S2,H2} + f_{S3,H2} = f_{H2,C1} + f_{H2,C2} + f_{H2,C3} + f_{H2,C4}$

4. **Hub throughput capacity** (for each $h \in H$):
   $$
   \sum_{s \in S} f_{s h} \leq u_h
   $$
   - For $H1$: $f_{S1,H1} + f_{S2,H1} + f_{S3,H1} \leq 170$
   - For $H2$: $f_{S1,H2} + f_{S2,H2} + f_{S3,H2} \leq 160$

5. **Nonnegativity**:
   $$
   f_{ij} \geq 0 \quad \text{for all arcs } (i,j) \text{ listed above}
   $$

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
& f_{ij} \geq 0 \quad \text{for all arcs } (i,j) \text{ listed above}
\end{align*}
\]