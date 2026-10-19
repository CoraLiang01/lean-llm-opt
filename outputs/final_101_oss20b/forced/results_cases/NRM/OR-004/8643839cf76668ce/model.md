##### Abstract Mathematical Model

**Sets**  
- \(S\): set of distribution centers, indexed by \(s\).  
- \(C\): set of customer groups, indexed by \(c\).

**Parameters**  
- \(d_c\): total demand of customer group \(c\).  
  *Source:* column **demand** in table **file_0_view_0**.  
- \(s_s\): supply capacity of distribution center \(s\).  
  *Source:* column **supply_capacity** in table **file_1_view_0**.  
- \(c_{s,c}\): transportation cost per unit from distribution center \(s\) to customer group \(c\).  
  *Source:* matrix **file_2_view_0** with row id column **Unnamed: 0** (distribution center) and column ids **C1**–**C12** (customer groups).

**Decision Variables**  
- \(x_{s,c}\): number of units shipped from distribution center \(s\) to customer group \(c\).  
  Domain: \(x_{s,c} \ge 0\) (continuous; integer not required by the problem statement).

**Objective Function**  
\[
\min \quad \sum_{s \in S}\sum_{c \in C} c_{s,c}\,x_{s,c}
\]

**Constraints**

1. **Supply Capacity Constraints**  
   \[
   \sum_{c \in C} x_{s,c} \;\le\; s_s, \quad \forall s \in S
   \]

2. **Demand Satisfaction Constraints**  
   \[
   \sum_{s \in S} x_{s,c} \;=\; d_c, \quad \forall c \in C
   \]

3. **Non‑negativity**  
   \[
   x_{s,c} \;\ge\; 0, \quad \forall s \in S,\; \forall c \in C
   \]

---

##### Data Mapping

| Parameter | Source Table ID | Source Column(s) |
|-----------|-----------------|------------------|
| \(d_c\) | **file_0_view_0** | **demand** |
| \(s_s\) | **file_1_view_0** | **supply_capacity** |
| \(c_{s,c}\) | **file_2_view_0** | **Unnamed: 0** (row id), **C1**–**C12** (column ids) |

The matrix relationship between distribution centers and customer groups is defined by the **file_2_view_0** table, with rows identified by **Unnamed: 0** and columns by **C1**–**C12**. No record values or counts are displayed; only the structure and mapping are provided.