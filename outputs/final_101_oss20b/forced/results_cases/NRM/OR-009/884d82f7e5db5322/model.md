**Abstract Model**

---

### Index Sets
- \(P\) – set of production plants  
- \(C\) – set of retail customers  

### Parameters
- \(d_c\) – daily demand of customer \(c \in C\)  
- \(s_p\) – production capacity of plant \(p \in P\)  
- \(c_{p,c}\) – transportation cost per unit from plant \(p\) to customer \(c\)  

### Decision Variables
- \(x_{p,c}\) – quantity of beverages shipped from plant \(p\) to customer \(c\)  

### Objective Function
$$
\min \sum_{p \in P}\sum_{c \in C} c_{p,c}\,x_{p,c}
$$

### Constraints
1. **Supply Capacity Constraints**  
   $$
   \sum_{c \in C} x_{p,c} \;\le\; s_p , \qquad \forall p \in P
   $$

2. **Demand Satisfaction Constraints**  
   $$
   \sum_{p \in P} x_{p,c} \;=\; d_c , \qquad \forall c \in C
   $$

3. **Non‑negativity**  
   $$
   x_{p,c} \;\ge\; 0 , \qquad \forall p \in P,\; \forall c \in C
   $$

---

### Data Mapping

| Parameter | Source Table ID | Source Column |
|-----------|-----------------|---------------|
| \(d_c\)   | `file_0_view_0` | `demand`      |
| \(s_p\)   | `file_1_view_0` | `supply_capacity` |
| \(c_{p,c}\) | `file_2_view_0` | `C1`, `C2`, `C3`, `C4` (columns corresponding to customers) |
| Plant ID  | `file_1_view_0` | `Unnamed: 0`  |
| Customer ID | `file_0_view_0` | `customer`   |

*No record values or counts are reproduced; only the table identifiers and column names are shown.*