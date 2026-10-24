**Transshipment Model**

Let  

* \(S=\{s\mid \text{NodeType}(s)=\text{SourceSupply}\}\)  
* \(C=\{c\mid \text{NodeType}(c)=\text{CustomerDemand}\}\)  
* \(H=\{h\mid h\text{ appears in } \text{hub\_capacity.csv}\}\)

Parameters  

* \( \text{Supply}_s \) – available supply at source \(s\) (from *node_supply_demand.csv*)  
* \( \text{Demand}_c \) – required demand at customer \(c\) (from *node_supply_demand.csv*)  
* \( \text{Cap}_h \) – throughput capacity of hub \(h\) (from *hub_capacity.csv*)  
* \( c_{ij} \) – unit transportation cost on arc \(i\!\to\!j\) (from *arc_costs.csv*)

Decision variables  

* \( f_{ij}\ge 0 \) – shipment flow on directed arc \(i\!\to\!j\)  
  (for all \((i,j)\) listed in *arc_costs.csv*)

---

\[
\begin{aligned}
\min\;&\sum_{(i,j)\in A} c_{ij}\,f_{ij} \\[4pt]
\text{s.t.}\;&
\sum_{h\in H} f_{s h}\;\le\;\text{Supply}_s
&&\forall\,s\in S \quad\text{(source supply)}\\[4pt]
&\sum_{h\in H} f_{h c}\;=\;\text{Demand}_c
&&\forall\,c\in C \quad\text{(customer demand)}\\[4pt]
&\sum_{s\in S} f_{s h}\;=\;\sum_{c\in C} f_{h c}
&&\forall\,h\in H \quad\text{(hub flow balance)}\\[4pt]
&\sum_{s\in S} f_{s h}\;\le\;\text{Cap}_h
&&\forall\,h\in H \quad\text{(hub throughput)}\\[4pt]
&f_{ij}\;\ge\;0
&&\forall\,(i,j)\in A
\end{aligned}
\]

---

### Data Mapping

| Table ID | Source File | Relevant Columns | Mapping |
|----------|-------------|------------------|---------|
| `file_0_view_0` | `node_supply_demand.csv` | `Node`, `NodeType`, `Amount` | `S = {Node | NodeType='SourceSupply'}`<br>`C = {Node | NodeType='CustomerDemand'}`<br>`Supply_s = Amount` for sources<br>`Demand_c = Amount` for customers |
| `file_1_view_0` | `hub_capacity.csv` | `Hub`, `ThroughputCapacity` | `H = {Hub}`<br>`Cap_h = ThroughputCapacity` |
| `file_2_view_0` | `arc_costs.csv` | `From`, `To`, `Cost` | `A = {(From,To)}`<br>`c_{ij} = Cost` |

All parameters are taken directly from the corresponding rows; no additional filtering or aggregation is applied beyond the node type distinctions. The model uses the full set of arcs listed in *arc_costs.csv* and respects the supply, demand, flow‑balance, and capacity constraints as specified.