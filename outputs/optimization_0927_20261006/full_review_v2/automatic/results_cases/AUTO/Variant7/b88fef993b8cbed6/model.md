##### Sets

- Sources: $S = \{S1, S2, S3\}$
- Hubs: $H = \{H1, H2\}$
- Customers: $C = \{C1, C2, C3, C4\}$
- Arcs: All $(i,j)$ pairs listed below with costs.

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
- Arc costs $c_{ij}$:
  - $c_{S1,H1} = 2$
  - $c_{S1,H2} = 6$
  - $c_{S2,H1} = 4$
  - $c_{S2,H2} = 3$
  - $c_{S3,H1} = 7$
  - $c_{S3,H2} = 2$
  - $c_{H1,C1} = 3$
  - $c_{H1,C2} = 4$
  - $c_{H1,C3} = 7$
  - $c_{H1,C4} = 8$
  - $c_{H2,C1} = 8$
  - $c_{H2,C2} = 6$
  - $c_{H2,C3} = 3$
  - $c_{H2,C4} = 4$

##### Decision Variables

- $f_{ij} \geq 0$ (continuous): shipment flow on arc $(i,j)$ for all arcs listed above.

##### Objective

Minimize total transportation cost:
$$
\min \sum_{(i,j)} c_{ij} f_{ij}
$$
where the sum is over all arcs listed above.

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

3. **Flow balance at each hub** (for each $h \in H$):
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
   f_{ij} \geq 0 \quad \text{for all arcs } (i,j) \text{ listed above}
   $$

##### Arc List (with costs)

- $f_{S1,H1}$ ($c=2$), $f_{S1,H2}$ ($c=6$)
- $f_{S2,H1}$ ($c=4$), $f_{S2,H2}$ ($c=3$)
- $f_{S3,H1}$ ($c=7$), $f_{S3,H2}$ ($c=2$)
- $f_{H1,C1}$ ($c=3$), $f_{H1,C2}$ ($c=4$), $f_{H1,C3}$ ($c=7$), $f_{H1,C4}$ ($c=8$)
- $f_{H2,C1}$ ($c=8$), $f_{H2,C2}$ ($c=6$), $f_{H2,C3}$ ($c=3$), $f_{H2,C4}$ ($c=4$)