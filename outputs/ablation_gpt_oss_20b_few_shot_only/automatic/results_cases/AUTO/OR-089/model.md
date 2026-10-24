**Symbolic Model**

Let  

* \(I=\{SC1,SC2,\dots ,SC10\}\) – set of candidate service centres  
* \(J=\{C1,C2,\dots ,C15\}\) – set of customers  

Parameters  

* \(f_i\) : fixed opening cost of centre \(i\)  \(i\in I\)  
* \(c_{ij}\) : per‑customer service cost of assigning customer \(j\) to centre \(i\) \(i\in I,\;j\in J\)

Decision variables  

* \(y_i \in \{0,1\}\) : 1 if centre \(i\) is opened, 0 otherwise  
* \(a_{ij} \in \{0,1\}\) : 1 if customer \(j\) is assigned to centre \(i\), 0 otherwise  

**Objective**

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,a_{ij}
\]

**Constraints**

1. **Customer assignment** – each customer is served by exactly one centre  

\[
\sum_{i\in I} a_{ij}=1 ,\qquad \forall j\in J
\]

2. **Centre activation** – a customer can be assigned only to an opened centre  

\[
a_{ij}\le y_i ,\qquad \forall i\in I,\; j\in J
\]

3. **Capacity limit** – an opened centre can serve at most 4 customers  

\[
\sum_{j\in J} a_{ij}\le 4\,y_i ,\qquad \forall i\in I
\]

4. **Variable domains**  

\[
y_i\in\{0,1\},\; a_{ij}\in\{0,1\},\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source File | Key Column | Value Column | Mapping Rule |
|-----------|-------------|------------|--------------|--------------|
| \(f_i\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv` | `Service Center Fixed Opening Cost` | `SC1`–`SC10` (row identifiers) | For each row with identifier `SCk`, set \(f_{SCk}\) to the numeric value in that row. |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv` | `Customer` | `SC1`–`SC10` (column headers) | For each row with identifier `Cj` and each column `SCi`, set \(c_{SCi,Cj}\) to the numeric value in that cell. |

No other data or constraints are required. The model above fully captures the problem as described.