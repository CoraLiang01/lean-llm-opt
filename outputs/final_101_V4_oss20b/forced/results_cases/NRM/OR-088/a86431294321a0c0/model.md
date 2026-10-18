##### Sets  
- \(I = \{1,2,\dots ,15\}\) (plant indices, corresponding to F1–F15)  
- \(J = \{1,2,\dots ,15\}\) (customer indices, corresponding to C1–C15)

##### Parameters  
- \(fixed\_cost_i\) (plant opening cost for plant \(i\))  
- \(capacity_i\) (maximum units plant \(i\) can supply)  
- \(transport\_cost_{ij}\) (per‑unit shipping cost from plant \(i\) to customer \(j\))  
- \(demand_j\) (total demand of customer \(j\))

##### Decision Variables  
- \(y_i \in \{0,1\}\) (1 if plant \(i\) is opened, 0 otherwise)  
- \(x_{ij} \ge 0\) (units shipped from plant \(i\) to customer \(j\))

##### Objective Function  
\[
\min \;\sum_{i\in I} fixed\_cost_i \, y_i \;+\; \sum_{i\in I}\sum_{j\in J} transport\_cost_{ij}\, x_{ij}
\]

##### Constraints  

###### 1. Demand Satisfaction  
\[
\sum_{i\in I} x_{ij} \;=\; demand_j , \qquad \forall j\in J
\]

###### 2. Plant Capacity (only if plant is opened)  
\[
\sum_{j\in J} x_{ij} \;\le\; capacity_i \, y_i , \qquad \forall i\in I
\]

###### 3. Non‑negativity  
\[
x_{ij} \;\ge\; 0 , \qquad \forall i\in I,\; j\in J
\]

###### 4. Binary Opening Decision  
\[
y_i \;\in\; \{0,1\} , \qquad \forall i\in I
\]

##### Data Mapping  
| Source | Table ID | Column Names |
|--------|----------|--------------|
| Plant data | `cost` | `plant`, `fixed_cost`, `capacity`, `C1`, `C2`, `C3`, `C4`, `C5`, `C6`, `C7`, `C8`, `C9`, `C10`, `C11`, `C12`, `C13`, `C14`, `C15` |
| Customer demand | `demand` | `customer`, `demand` |

The model above uses the exact parameters extracted from the `cost.csv` and `demand.csv` files as provided by the CSVQA observation.