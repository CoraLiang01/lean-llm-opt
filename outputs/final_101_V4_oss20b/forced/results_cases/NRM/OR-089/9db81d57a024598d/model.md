##### Sets  

\(I=\{SC1,SC2,SC3,SC4,SC5,SC6,SC7,SC8,SC9,SC10\}\)  

\(J=\{C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15\}\)  

##### Parameters  

\(f_i\) – fixed opening cost for centre \(i\) (from column **Fixed Opening Cost** in table **service_centers_fixed_costs.csv**)  

\(c_{ij}\) – service cost for customer \(j\) from centre \(i\) (from column **SCi** in table **expanded_customer_service_costs.csv**)  

##### Decision Variables  

\(y_i\in\{0,1\}\) indicator that centre \(i\) is opened  

\(x_{ij}\in\{0,1\}\) indicator that customer \(j\) is assigned to centre \(i\)  

##### Objective Function  

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

##### Constraints  

1. **Customer assignment**  
\[
\sum_{i\in I} x_{ij}=1 \quad \forall j\in J
\]

2. **Assignment only to opened centres**  
\[
x_{ij}\le y_i \quad \forall i\in I,\;\forall j\in J
\]

3. **Capacity of opened centres**  
\[
\sum_{j\in J} x_{ij}\le 4\,y_i \quad \forall i\in I
\]

4. **Variable domains**  
\[
y_i\in\{0,1\}\quad \forall i\in I
\]  
\[
x_{ij}\in\{0,1\}\quad \forall i\in I,\;\forall j\in J
\]

##### Data Mapping  

| Parameter | Source Table | Exact Column Name |
|-----------|--------------|-------------------|
| \(f_i\) | **service_centers_fixed_costs.csv** | **Fixed Opening Cost** |
| \(c_{ij}\) | **expanded_customer_service_costs.csv** | **SC1**, **SC2**, **SC3**, **SC4**, **SC5**, **SC6**, **SC7**, **SC8**, **SC9**, **SC10** |

All indices \(i\) and \(j\) correspond to the values in the **Service Center** and **Customer** columns of the respective tables.