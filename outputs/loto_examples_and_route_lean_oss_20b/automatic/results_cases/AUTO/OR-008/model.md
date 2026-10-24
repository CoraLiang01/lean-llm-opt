## Mathematical Model

**Sets**  
\(S=\{Supplier1,Supplier2,Supplier3,Supplier4,Supplier5\}\)  
\(C=\{Customer1,Customer2,Customer3,Customer4,Customer5,Customer6\}\)

**Parameters**  
\(d_c\) – demand of customer \(c\) (from *file_0_view_0*, column *demand*)  
\(s_s\) – supply capacity of supplier \(s\) (from *file_1_view_0*, column *supply_capacity*)  
\(c_{s,c}\) – transportation cost per unit from supplier \(s\) to customer \(c\) (from *file_2_view_0*, matrix entry at row *Unnamed: 0* = \(s\), column \(c\))

**Decision Variables**  
\(x_{s,c}\ge 0\) – units shipped from supplier \(s\) to customer \(c\)

**Objective**  
\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

**Constraints**

*Demand satisfaction*  
\[
\sum_{s\in S} x_{s,c}=d_c \qquad \forall\,c\in C
\]

*Supply capacity*  
\[
\sum_{c\in C} x_{s,c}\le s_s \qquad \forall\,s\in S
\]

*Non‑negativity*  
\[
x_{s,c}\ge 0 \qquad \forall\,s\in S,\;c\in C
\]

---

## Data Mapping

| Parameter | Source Table | Source Column | Mapping |
|-----------|--------------|---------------|---------|
| \(d_c\) | *file_0_view_0* | *demand* | row identified by *Customers* |
| \(s_s\) | *file_1_view_0* | *supply_capacity* | row identified by *Suppliers* |
| \(c_{s,c}\) | *file_2_view_0* | matrix entry | row identified by *Unnamed: 0* = supplier, column = customer |

All indices \(s\) and \(c\) are taken from the full set of rows returned in the observation; no additional filtering is applied.