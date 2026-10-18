##### Abstract Model

**Index Sets**  
- \(S\): set of Walmart stores (indexed by the store identifier column in `supply_capacity.csv`).  
- \(C\): set of customer groups (indexed by the customer identifier column in `customer_demand.csv`).

**Parameters**  
- \(d_c\): total demand of customer group \(c \in C\).  
- \(s_s\): supply capacity of store \(s \in S\).  
- \(c_{s,c}\): transportation cost per unit of goods from store \(s\) to customer \(c\).

**Decision Variables**  
- \(x_{s,c}\): quantity of goods transported from store \(s\) to customer \(c\).  
  \(x_{s,c} \in \mathbb{Z}_{\ge 0}\).

**Objective Function**  
\[
\min \sum_{s \in S}\sum_{c \in C} c_{s,c}\,x_{s,c}
\]

**Constraints**

1. **Demand Satisfaction**  
\[
\sum_{s \in S} x_{s,c} \;\ge\; d_c , \qquad \forall c \in C
\]

2. **Supply Capacity**  
\[
\sum_{c \in C} x_{s,c} \;\le\; s_s , \qquad \forall s \in S
\]

3. **Non‑negativity and Integer Domain**  
\[
x_{s,c} \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall s \in S,\; \forall c \in C
\]

---

##### Data Mapping

| Parameter | Source Table ID | Source Column Name |
|-----------|-----------------|--------------------|
| \(d_c\)   | `file_0_view_0` | `demand` |
| \(s_s\)   | `file_1_view_0` | `supply_capacity` |
| \(c_{s,c}\) | `file_2_view_0` | `C1`, `C2`, …, `C12` (one column per customer) |

*Relationships:*  
- The `transportation_costs.csv` table (`file_2_view_0`) contains a row identifier column `Unnamed: 0` that matches the store identifiers in `supply_capacity.csv` (`file_1_view_0`).  
- Each column `C1`–`C12` in `file_2_view_0` corresponds to a customer identifier that matches the `customer_id` column in `customer_demand.csv` (`file_0_view_0`).