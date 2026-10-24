**Mathematical Model**

Let  

* \(S=\{SC1,SC2,\dots ,SC10\}\) – set of candidate service centres  
* \(C=\{C1,C2,\dots ,C15\}\) – set of customers  

Parameters  

* \(f_s\) – fixed opening cost of centre \(s\)  
  \(\displaystyle f_s = \text{“Fixed Opening Cost” from table\_id\_file\_1\_view\_0 for Service Center }s\)  

* \(c_{s,c}\) – per‑customer service cost of assigning customer \(c\) to centre \(s\)  
  \(\displaystyle c_{s,c} = \text{“SC\(s\)” column from table\_id\_file\_0\_view\_0 for Customer }c\)

Decision variables  

* \(y_s \in \{0,1\}\) – 1 if centre \(s\) is opened, 0 otherwise  
* \(x_{s,c} \in \{0,1\}\) – 1 if customer \(c\) is served by centre \(s\), 0 otherwise  

Objective  

\[
\min \;\; \sum_{s\in S} f_s\,y_s \;+\; \sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

Constraints  

1. **Customer assignment** – each customer is served exactly once  

\[
\sum_{s\in S} x_{s,c}=1 \quad \forall\,c\in C
\]

2. **Assignment only to opened centres**  

\[
x_{s,c}\le y_s \quad \forall\,s\in S,\;c\in C
\]

3. **Centre capacity** – at most 4 customers per opened centre  

\[
\sum_{c\in C} x_{s,c}\le 4\,y_s \quad \forall\,s\in S
\]

4. **Variable domains**  

\[
y_s\in\{0,1\}\quad \forall\,s\in S,\qquad
x_{s,c}\in\{0,1\}\quad \forall\,s\in S,\;c\in C
\]

---

### Data Mapping

| Parameter | Source | Table ID | Column(s) | Mapping |
|-----------|--------|----------|-----------|---------|
| \(f_s\) | Fixed opening costs | `file_1_view_0` | `Fixed Opening Cost` | `f_s = value where Service Center = s` |
| \(c_{s,c}\) | Customer‑centre service costs | `file_0_view_0` | `SC1`–`SC10` | `c_{s,c} = value in column SCs for row Customer = c` |

*Index sets*  
- \(S = \{SC1, SC2, \dots , SC10\}\) (all rows of `file_1_view_0`)  
- \(C = \{C1, C2, \dots , C15\}\) (all rows of `file_0_view_0`)

All parameters are taken directly from the corresponding CSV files; no additional data or assumptions are introduced. The model is a standard mixed‑integer linear program with binary variables and linear constraints.