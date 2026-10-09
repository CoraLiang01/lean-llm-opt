#### Sets

- Sources: $S = \{\text{S1}, \text{S2}, \text{S3}\}$
- Hubs: $H = \{\text{H1}, \text{H2}\}$
- Customers: $C = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$
- Arcs: All $(i,j)$ pairs listed below with associated costs.

#### Parameters

- Source supplies:
  - $s_{\text{S1}} = 120$
  - $s_{\text{S2}} = 100$
  - $s_{\text{S3}} = 90$
- Customer demands:
  - $d_{\text{C1}} = 70$
  - $d_{\text{C2}} = 80$
  - $d_{\text{C3}} = 60$
  - $d_{\text{C4}} = 90$
- Hub throughput capacities:
  - $u_{\text{H1}} = 170$
  - $u_{\text{H2}} = 160$
- Arc costs $c_{ij}$ (for each directed arc $(i,j)$):

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

#### Decision Variables

- $f_{ij} \geq 0$: shipment flow on arc $(i,j)$, for all arcs listed above.

#### Objective

Minimize total transportation cost:
$$
\min \sum_{(i,j)} c_{ij} f_{ij}
$$
where the sum is over all arcs listed above.

#### Constraints

1. **Source supply upper bounds** (for each source $s \in S$):
   $$
   \sum_{h \in H} f_{s h} \leq s_s
   $$
   - For S1: $f_{\text{S1,H1}} + f_{\text{S1,H2}} \leq 120$
   - For S2: $f_{\text{S2,H1}} + f_{\text{S2,H2}} \leq 100$
   - For S3: $f_{\text{S3,H1}} + f_{\text{S3,H2}} \leq 90$

2. **Customer demand constraints** (for each customer $c \in C$):
   $$
   \sum_{h \in H} f_{h c} \geq d_c
   $$
   - For C1: $f_{\text{H1,C1}} + f_{\text{H2,C1}} \geq 70$
   - For C2: $f_{\text{H1,C2}} + f_{\text{H2,C2}} \geq 80$
   - For C3: $f_{\text{H1,C3}} + f_{\text{H2,C3}} \geq 60$
   - For C4: $f_{\text{H1,C4}} + f_{\text{H2,C4}} \geq 90$

3. **Flow balance at each hub** (for each $h \in H$):
   $$
   \sum_{s \in S} f_{s h} = \sum_{c \in C} f_{h c}
   $$
   - For H1: $f_{\text{S1,H1}} + f_{\text{S2,H1}} + f_{\text{S3,H1}} = f_{\text{H1,C1}} + f_{\text{H1,C2}} + f_{\text{H1,C3}} + f_{\text{H1,C4}}$
   - For H2: $f_{\text{S1,H2}} + f_{\text{S2,H2}} + f_{\text{S3,H2}} = f_{\text{H2,C1}} + f_{\text{H2,C2}} + f_{\text{H2,C3}} + f_{\text{H2,C4}}$

4. **Hub throughput capacity** (for each $h \in H$):
   $$
   \sum_{s \in S} f_{s h} \leq u_h
   $$
   - For H1: $f_{\text{S1,H1}} + f_{\text{S2,H1}} + f_{\text{S3,H1}} \leq 170$
   - For H2: $f_{\text{S1,H2}} + f_{\text{S2,H2}} + f_{\text{S3,H2}} \leq 160$

5. **Nonnegativity**:
   $$
   f_{ij} \geq 0 \quad \text{for all arcs } (i,j)
   $$

#### Arc List and Costs

| Arc           | Cost |
|---------------|------|
| S1 $\to$ H1   | 2    |
| S1 $\to$ H2   | 6    |
| S2 $\to$ H1   | 4    |
| S2 $\to$ H2   | 3    |
| S3 $\to$ H1   | 7    |
| S3 $\to$ H2   | 2    |
| H1 $\to$ C1   | 3    |
| H1 $\to$ C2   | 4    |
| H1 $\to$ C3   | 7    |
| H1 $\to$ C4   | 8    |
| H2 $\to$ C1   | 8    |
| H2 $\to$ C2   | 6    |
| H2 $\to$ C3   | 3    |
| H2 $\to$ C4   | 4    |