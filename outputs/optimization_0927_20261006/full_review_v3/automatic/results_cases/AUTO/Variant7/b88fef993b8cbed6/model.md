##### Sets

- Sources: $S = \{S1, S2, S3\}$
- Hubs: $H = \{H1, H2\}$
- Customers: $C = \{C1, C2, C3, C4\}$

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
- Arc costs (all arcs with positive cost):
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

- $f_{ij} \geq 0$ (continuous): shipment flow on arc $i \to j$ for each arc listed above.

##### Objective

Minimize total transportation cost:
$$
\min \Bigg[
\begin{aligned}
&2f_{S1,H1} + 6f_{S1,H2} + 4f_{S2,H1} + 3f_{S2,H2} + 7f_{S3,H1} + 2f_{S3,H2} \\
&+ 3f_{H1,C1} + 4f_{H1,C2} + 7f_{H1,C3} + 8f_{H1,C4} \\
&+ 8f_{H2,C1} + 6f_{H2,C2} + 3f_{H2,C3} + 4f_{H2,C4}
\end{aligned}
\Bigg]
$$

##### Constraints

1. **Source supply upper bounds** (for each $s \in S$):
   $$
   \begin{aligned}
   &f_{S1,H1} + f_{S1,H2} \leq 120 \\
   &f_{S2,H1} + f_{S2,H2} \leq 100 \\
   &f_{S3,H1} + f_{S3,H2} \leq 90
   \end{aligned}
   $$

2. **Customer demand constraints** (for each $c \in C$):
   $$
   \begin{aligned}
   &f_{H1,C1} + f_{H2,C1} \geq 70 \\
   &f_{H1,C2} + f_{H2,C2} \geq 80 \\
   &f_{H1,C3} + f_{H2,C3} \geq 60 \\
   &f_{H1,C4} + f_{H2,C4} \geq 90
   \end{aligned}
   $$

3. **Flow balance at each hub** (for each $h \in H$):
   $$
   \begin{aligned}
   &f_{S1,H1} + f_{S2,H1} + f_{S3,H1} = f_{H1,C1} + f_{H1,C2} + f_{H1,C3} + f_{H1,C4} \\
   &f_{S1,H2} + f_{S2,H2} + f_{S3,H2} = f_{H2,C1} + f_{H2,C2} + f_{H2,C3} + f_{H2,C4}
   \end{aligned}
   $$

4. **Hub throughput capacity constraints** (for each $h \in H$):
   $$
   \begin{aligned}
   &f_{S1,H1} + f_{S2,H1} + f_{S3,H1} \leq 170 \\
   &f_{S1,H2} + f_{S2,H2} + f_{S3,H2} \leq 160
   \end{aligned}
   $$

5. **Nonnegativity** (for all arcs listed above):
   $$
   f_{ij} \geq 0
   $$

##### Variables (all continuous and nonnegative):

- $f_{S1,H1},\ f_{S1,H2},\ f_{S2,H1},\ f_{S2,H2},\ f_{S3,H1},\ f_{S3,H2}$
- $f_{H1,C1},\ f_{H1,C2},\ f_{H1,C3},\ f_{H1,C4}$
- $f_{H2,C1},\ f_{H2,C2},\ f_{H2,C3},\ f_{H2,C4}$

##### Data (retrieved, source order):

- node_supply_demand.csv:
  - S1, SourceSupply, 120
  - S2, SourceSupply, 100
  - S3, SourceSupply, 90
  - C1, CustomerDemand, 70
  - C2, CustomerDemand, 80
  - C3, CustomerDemand, 60
  - C4, CustomerDemand, 90
- hub_capacity.csv:
  - H1, 170
  - H2, 160
- arc_costs.csv:
  - S1,H1,2
  - S1,H2,6
  - S2,H1,4
  - S2,H2,3
  - S3,H1,7
  - S3,H2,2
  - H1,C1,3
  - H1,C2,4
  - H1,C3,7
  - H1,C4,8
  - H2,C1,8
  - H2,C2,6
  - H2,C3,3
  - H2,C4,4