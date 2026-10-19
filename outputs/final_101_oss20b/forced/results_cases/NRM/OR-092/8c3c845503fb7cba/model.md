##### Sets
- \(S = \{s \mid s \in \{S1,S2,\dots ,S10\}\}\)
- \(D = \{d \mid d \in \{D1,D2,\dots ,D20\}\}\)

##### Parameters
- \(\text{supply}_s \in \mathbb{R}\) (available units at source \(s\), from table **file_2_view_0**, column **supply_units**)
- \(\text{demand}_d \in \mathbb{R}\) (required units at destination \(d\), from table **file_1_view_0**, column **demand_units**)
- \(\text{cost}_{sd} \in \mathbb{R}\) (unit transportation cost from \(s\) to \(d\), from table **file_0_view_0**, columns **D1**–**D20**)
- \(\text{truck\_capacity} = 10\) (units per truck)

##### Decision Variables
- \(x_{sd} \in \mathbb{Z}_{\ge 0}\) (number of trucks dispatched from source \(s\) to destination \(d\))

##### Objective Function
\[
\min Z = \sum_{s \in S}\sum_{d \in D} \text{cost}_{sd}\;(\text{truck\_capacity}\;x_{sd})
\]

##### Constraints
1. **Demand satisfaction**  
   \[
   \sum_{s \in S} \text{truck\_capacity}\;x_{sd} = \text{demand}_d , \quad \forall d \in D
   \]
2. **Supply limitation**  
   \[
   \sum_{d \in D} \text{truck\_capacity}\;x_{sd} \le \text{supply}_s , \quad \forall s \in S
   \]
3. **Integer non‑negativity**  
   \[
   x_{sd} \in \mathbb{Z}_{\ge 0}, \quad \forall s \in S,\; d \in D
   \]

##### Data Mapping
| Parameter | Source Table ID | Column Name |
|-----------|-----------------|-------------|
| \(\text{supply}_s\) | **file_2_view_0** | **supply_units** |
| \(\text{demand}_d\) | **file_1_view_0** | **demand_units** |
| \(\text{cost}_{sd}\) | **file_0_view_0** | **D1**–**D20** (one column per destination) |
| \(\text{truck\_capacity}\) | — | constant value 10 |

This abstract model captures the transportation problem described, using the supplied CSV data without including any specific record values.