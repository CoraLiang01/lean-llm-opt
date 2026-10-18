##### Model

**Index Sets**  
- \(S\) : set of suppliers  
- \(C\) : set of customer groups  

**Parameters**  
- \(\text{Supply}[s]\) : daily supply capacity of supplier \(s\) (from *supply_capacity.csv*)  
- \(\text{Demand}[c]\) : daily demand of customer group \(c\) (from *customer_demand.csv*)  
- \(\text{Cost}[s,c]\) : transportation cost per unit from supplier \(s\) to customer \(c\) (from *transportation_costs.csv*)  

**Decision Variables**  
- \(x_{s,c} \in \mathbb{Z}_{\ge 0}\) : number of units shipped from supplier \(s\) to customer \(c\)  

**Objective Function**  
\[
\min \quad \sum_{s \in S}\sum_{c \in C} \text{Cost}[s,c] \cdot x_{s,c}
\]

**Constraints**

1. **Supply Capacity Constraints**  
\[
\sum_{c \in C} x_{s,c} \;\le\; \text{Supply}[s], \quad \forall s \in S
\]

2. **Demand Satisfaction Constraints**  
\[
\sum_{s \in S} x_{s,c} \;=\; \text{Demand}[c], \quad \forall c \in C
\]

3. **Variable Domain**  
\[
x_{s,c} \in \mathbb{Z}_{\ge 0}, \quad \forall s \in S,\; \forall c \in C
\]

##### Data Mapping

| Source Table | Relevant Columns | Description |
|--------------|------------------|-------------|
| *supply_capacity.csv* | `Unnamed: 0`, `supply_capacity` | Maps to \(S\) and \(\text{Supply}[s]\) |
| *customer_demand.csv* | `customer`, `demand` | Maps to \(C\) and \(\text{Demand}[c]\) |
| *transportation_costs.csv* | `Unnamed: 0`, `C1`–`C10` | Maps to \(\text{Cost}[s,c]\) |

*No record values or counts are reproduced; only column identifiers are shown.*