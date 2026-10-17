**Mathematical Model**

Let  

- \(C=\{C_1,\dots ,C_{15}\}\) – set of customers  
- \(S=\{SC_1,\dots ,SC_{10}\}\) – set of candidate service centres  

Parameters  

- \(f_s\) – fixed opening cost of centre \(s\)  
- \(c_{s,c}\) – service cost of assigning customer \(c\) to centre \(s\)  

Decision variables  

- \(y_s \in \{0,1\}\) – 1 if centre \(s\) is opened, 0 otherwise  
- \(x_{s,c} \in \{0,1\}\) – 1 if customer \(c\) is served by centre \(s\), 0 otherwise  

**Objective**

\[
\min \;\; \sum_{s\in S} f_s\,y_s \;+\; \sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

**Constraints**

1. **Customer assignment**  
   \[
   \sum_{s\in S} x_{s,c}=1 \quad \forall\,c\in C
   \]

2. **Assignment only to open centres**  
   \[
   x_{s,c}\le y_s \quad \forall\,s\in S,\;c\in C
   \]

3. **Capacity of opened centres**  
   \[
   \sum_{c\in C} x_{s,c}\le 4\,y_s \quad \forall\,s\in S
   \]

4. **Variable domains**  
   \[
   y_s\in\{0,1\}\;\;\forall s\in S,\qquad
   x_{s,c}\in\{0,1\}\;\;\forall s\in S,\;c\in C
   \]

---

**Data Mapping**

| Parameter | Source Table ID | Source Column | Key Column |
|-----------|-----------------|---------------|------------|
| \(f_s\)   | `file_1_view_0` | `Fixed Opening Cost` | `Service Center` |
| \(c_{s,c}\) | `file_0_view_0` | `SC1`–`SC10` | `Customer` |

All parameter values are taken directly from the corresponding CSV files; no additional data or transformations are introduced.