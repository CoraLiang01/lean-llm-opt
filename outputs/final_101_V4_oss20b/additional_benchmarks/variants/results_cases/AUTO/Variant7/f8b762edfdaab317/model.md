**Transshipment Model**

Let  

* \(S=\{s\}\) – source plants (from *node_supply_demand.csv* where *NodeType* = *SourceSupply*)  
* \(H=\{h\}\) – cross‑dock hubs (from *hub_capacity.csv*)  
* \(C=\{c\}\) – customers (from *node_supply_demand.csv* where *NodeType* = *CustomerDemand*)  
* \(A=\{(i,j)\}\) – directed arcs listed in *arc_costs.csv*  

Parameters  

\[
\begin{aligned}
&\text{Supply}_s &&\ge 0 &&\text{(source supply)}\\
&\text{Demand}_c &&\ge 0 &&\text{(customer demand)}\\
&\text{Cap}_h &&\ge 0 &&\text{(hub throughput capacity)}\\
&\text{Cost}_{ij} &&\ge 0 &&\text{(per‑unit transport cost)}
\end{aligned}
\]

Decision variables  

\[
f_{ij}\;\ge\;0 \qquad (i,j)\in A
\]

---

### Objective  

\[
\min \;\sum_{(i,j)\in A}\text{Cost}_{ij}\,f_{ij}
\]

### Constraints  

1. **Source supply limits**  
   \[
   \sum_{j\in H} f_{sj}\;\le\;\text{Supply}_s \qquad \forall s\in S
   \]

2. **Customer demand satisfaction**  
   \[
   \sum_{i\in H} f_{ic}\;\ge\;\text{Demand}_c \qquad \forall c\in C
   \]

3. **Hub flow balance**  
   \[
   \sum_{i\in S} f_{ih}\;=\;\sum_{j\in C} f_{hj} \qquad \forall h\in H
   \]

4. **Hub throughput capacity**  
   \[
   \sum_{i\in S} f_{ih}\;\le\;\text{Cap}_h \qquad \forall h\in H
   \]

5. **Non‑negativity**  
   \[
   f_{ij}\;\ge\;0 \qquad \forall (i,j)\in A
   \]

---

## Data Mapping Rules

| Parameter | Source Table | Key Columns | Value Column | Notes |
|-----------|--------------|-------------|--------------|-------|
| `Supply_s` | `file_0_view_0` | `Node` (where `NodeType` = `SourceSupply`) | `Amount` | Source supply |
| `Demand_c` | `file_0_view_0` | `Node` (where `NodeType` = `CustomerDemand`) | `Amount` | Customer demand |
| `Cap_h` | `file_1_view_0` | `Hub` | `ThroughputCapacity` | Hub capacity |
| `Cost_{ij}` | `file_2_view_0` | `From`, `To` | `Cost` | Per‑unit cost for each directed arc |

*All identifiers (`S`, `H`, `C`, `A`) are taken directly from the corresponding tables; no additional aggregation or renaming is performed.*