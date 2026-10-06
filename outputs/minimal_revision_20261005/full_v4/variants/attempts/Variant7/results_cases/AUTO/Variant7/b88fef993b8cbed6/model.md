##### Decision Variables

For each directed arc $(i,j)$ listed in arc_costs.csv, let
$$
f_{ij} \geq 0
$$
be the nonnegative continuous shipment flow on arc $i \to j$.

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{(i,j) \in \mathcal{A}} c_{ij} f_{ij}
$$
where $\mathcal{A}$ is the set of all arcs in arc_costs.csv, and $c_{ij}$ is the per-unit cost from column Cost.

##### Constraints

1. **Source Supply Upper Bounds**  
   For each source $s$ with supply $S_s$ (NodeType = SourceSupply in node_supply_demand.csv):
   $$
   \sum_{j: (s,j) \in \mathcal{A}} f_{sj} \leq S_s
   $$
   where $S_s$ is Amount for Node $s$.

2. **Customer Demand Satisfaction**  
   For each customer $c$ with demand $D_c$ (NodeType = CustomerDemand in node_supply_demand.csv):
   $$
   \sum_{i: (i,c) \in \mathcal{A}} f_{ic} \geq D_c
   $$
   where $D_c$ is Amount for Node $c$.

3. **Flow Balance at Each Hub**  
   For each hub $h$ (Hub in hub_capacity.csv):
   $$
   \sum_{i: (i,h) \in \mathcal{A}} f_{ih} = \sum_{j: (h,j) \in \mathcal{A}} f_{hj}
   $$

4. **Hub Throughput Capacity**  
   For each hub $h$ with throughput capacity $T_h$ (ThroughputCapacity in hub_capacity.csv):
   $$
   \sum_{i: (i,h) \in \mathcal{A}} f_{ih} \leq T_h
   $$

5. **Nonnegativity**  
   $$
   f_{ij} \geq 0 \qquad \forall (i,j) \in \mathcal{A}
   $$

---

#### Data Mapping

- **Source nodes and supplies** (from node_supply_demand.csv, table_id: file_0_view_0, NodeType = SourceSupply):  
  - $S_1$: $S_{S1} = 120$  
  - $S_2$: $S_{S2} = 100$  
  - $S_3$: $S_{S3} = 90$  

- **Customer nodes and demands** (from node_supply_demand.csv, table_id: file_0_view_0, NodeType = CustomerDemand):  
  - $C_1$: $D_{C1} = 70$  
  - $C_2$: $D_{C2} = 80$  
  - $C_3$: $D_{C3} = 60$  
  - $C_4$: $D_{C4} = 90$  

- **Hubs and capacities** (from hub_capacity.csv, table_id: file_1_view_0):  
  - $H_1$: $T_{H1} = 170$  
  - $H_2$: $T_{H2} = 160$  

- **Arcs and costs** (from arc_costs.csv, table_id: file_2_view_0):  
  - All arcs $(i,j)$ with per-unit cost $c_{ij}$ as in columns From, To, Cost.

---

**Index sets:**
- Sources: $S = \{S1, S2, S3\}$
- Hubs: $H = \{H1, H2\}$
- Customers: $C = \{C1, C2, C3, C4\}$
- Arcs: $\mathcal{A} = \{(i,j):$ row in arc_costs.csv$\}$

---

**All parameters and sets are bound directly to the retrieved data as specified above.**