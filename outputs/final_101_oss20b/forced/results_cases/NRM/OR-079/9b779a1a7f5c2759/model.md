**Abstract Model**

**Index Sets**  
\(F = \{A1, A2, \dots , A15\}\) – set of potential factory sites  
\(D = \{B1, B2, \dots , B8\}\) – set of distribution centers  

**Parameters**  
\(f_f\) : fixed construction cost of factory \(f \in F\)  (from *facility_costs.csv*, column **FixedCost**)  
\(c_f\) : production capacity of factory \(f \in F\)  (from *facility_costs.csv*, column **Capacity**)  
\(c_{fd}\) : variable shipping cost per unit from factory \(f\) to distribution center \(d\) (from *shipping_costs.csv*, columns **B1**–**B8**)  
\(d_d\) : demand at distribution center \(d \in D\)  (from *demand_requirements.csv*, column **Demand**)  

**Decision Variables**  
\(y_f \in \{0,1\}\) : 1 if factory \(f\) is built, 0 otherwise  
\(x_{fd} \in \mathbb{Z}_{+}\) : number of units shipped from factory \(f\) to distribution center \(d\)  

**Objective Function**  
\[
\min \;\; \sum_{f \in F} f_f \, y_f \;+\; \sum_{f \in F}\sum_{d \in D} c_{fd}\, x_{fd}
\]

**Constraints**

1. **Capacity Constraints**  
\[
\sum_{d \in D} x_{fd} \;\le\; c_f \, y_f , \qquad \forall f \in F
\]

2. **Demand Satisfaction**  
\[
\sum_{f \in F} x_{fd} \;\ge\; d_d , \qquad \forall d \in D
\]

3. **Non‑negativity and Integrality**  
\[
x_{fd} \;\ge\; 0 , \quad x_{fd} \in \mathbb{Z}, \qquad \forall f \in F,\; d \in D
\]

4. **Binary Construction Decision**  
\[
y_f \in \{0,1\}, \qquad \forall f \in F
\]

---

### Data Mapping

| Parameter | Source Table | Table ID | Column Name |
|-----------|--------------|----------|-------------|
| \(f_f\) | *facility_costs.csv* | `file_0_view_0` | `FixedCost` |
| \(c_f\) | *facility_costs.csv* | `file_0_view_0` | `Capacity` |
| \(c_{fd}\) | *shipping_costs.csv* | `file_1_view_0` | `B1`–`B8` (one column per destination) |
| \(d_d\) | *demand_requirements.csv* | `file_2_view_0` | `Demand` |

All data are taken directly from the corresponding CSV files; no record values or counts are reproduced in the model.