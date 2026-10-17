**Mathematical Model**

Let  

* \(S=\{S1,\dots ,S18\}\) – set of distribution centers (suppliers).  
* \(C=\{C1,\dots ,C18\}\) – set of customer groups.  

Parameters  

* \(d_c\) – daily demand of customer \(c\) (from *file_0_view_0*, column **demand**).  
* \(cap_s\) – daily supply capacity of supplier \(s\) (from *file_1_view_0*, column **supply_capacity**).  
* \(c_{s,c}\) – transportation cost per unit from supplier \(s\) to customer \(c\) (from *file_2_view_0*, columns **C1**–**C18**).  

Decision variables  

* \(x_{s,c}\ge 0\) – quantity shipped from supplier \(s\) to customer \(c\) (continuous).  

**Objective**  
\[
\min \; Z = \sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

**Constraints**

*Demand satisfaction*  
\[
\sum_{s\in S} x_{s,c} = d_c \quad \forall\,c\in C
\]

*Supply capacity*  
\[
\sum_{c\in C} x_{s,c} \le cap_s \quad \forall\,s\in S
\]

*Non‑negativity*  
\[
x_{s,c}\ge 0 \quad \forall\,s\in S,\;c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Source Columns | Mapping |
|-----------|--------------|----------------|---------|
| \(d_c\) | *file_0_view_0* | **customer**, **demand** | \(d_c =\) value in column **demand** where **customer** = \(c\) |
| \(cap_s\) | *file_1_view_0* | **Unnamed: 0**, **supply_capacity** | \(cap_s =\) value in column **supply_capacity** where **Unnamed: 0** = \(s\) |
| \(c_{s,c}\) | *file_2_view_0* | **Unnamed: 0**, **C1**–**C18** | \(c_{s,c} =\) value in column **C\(c\)** where **Unnamed: 0** = \(s\) |

All indices \(s\) and \(c\) are taken from the full sets of rows in the respective tables; no additional filtering is applied.