**Mathematical Model**

Let  

* \(S=\{SC1,SC2,\dots ,SC10\}\) – set of candidate service centres  
* \(C=\{C1,C2,\dots ,C15\}\) – set of customers  

Parameters  

* \(f_s\) – fixed opening cost of centre \(s\) (from *service_centers_fixed_costs.csv*)  
* \(c_{s,c}\) – cost of serving customer \(c\) from centre \(s\) (from *expanded_customer_service_costs.csv*)  

Decision variables  

* \(y_s\in\{0,1\}\) – 1 if centre \(s\) is opened, 0 otherwise  
* \(x_{s,c}\in\{0,1\}\) – 1 if customer \(c\) is assigned to centre \(s\), 0 otherwise  

Objective  

\[
\min \;\; \sum_{s\in S} f_s\,y_s \;+\; \sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

Constraints  

1. **Each customer is served exactly once**  
\[
\sum_{s\in S} x_{s,c}=1 \qquad \forall\,c\in C
\]

2. **Assignment only to opened centres**  
\[
x_{s,c}\le y_s \qquad \forall\,s\in S,\;c\in C
\]

3. **Capacity of each centre**  
\[
\sum_{c\in C} x_{s,c}\le 4\,y_s \qquad \forall\,s\in S
\]

4. **Variable domains**  
\[
y_s\in\{0,1\}\quad \forall\,s\in S,\qquad
x_{s,c}\in\{0,1\}\quad \forall\,s\in S,\;c\in C
\]

---

### Data Mapping

| Parameter | Source Table | Columns | Mapping |
|-----------|--------------|---------|---------|
| \(f_s\) | *service_centers_fixed_costs.csv* (table_id `file_1_view_0`) | `Service Center`, `Fixed Opening Cost` | \(f_{SCi} =\) value in `Fixed Opening Cost` where `Service Center` = `SCi` |
| \(c_{s,c}\) | *expanded_customer_service_costs.csv* (table_id `file_0_view_0`) | `Customer`, `SC1`–`SC10` | \(c_{SCi,Cj} =\) value in column `SCi` where `Customer` = `Cj` |

*Index sets*  
- \(S\) is the set of all `Service Center` values in `file_1_view_0`.  
- \(C\) is the set of all `Customer` values in `file_0_view_0`.

All parameters are taken directly from the corresponding CSV files; no additional processing or filtering is applied. The model uses only the data present in these two tables.