**Mathematical Model**

Let  

* \(I=\{C1,\dots ,C15\}\) – set of customers  
* \(S=\{SC1,\dots ,SC10\}\) – set of candidate service centres  

Parameters  

* \(f_s\) – fixed opening cost of centre \(s\)  \(s\in S\)  
* \(c_{is}\) – per‑customer service cost of assigning customer \(i\) to centre \(s\) \(i\in I,\;s\in S\)

Decision variables  

* \(y_s\in\{0,1\}\) – 1 if centre \(s\) is opened, 0 otherwise  
* \(x_{is}\in\{0,1\}\) – 1 if customer \(i\) is served by centre \(s\), 0 otherwise  

**Objective**  
\[
\min \;\; \sum_{s\in S} f_s\,y_s \;+\; \sum_{i\in I}\sum_{s\in S} c_{is}\,x_{is}
\]

**Constraints**

1. **Each customer is served exactly once**  
   \[
   \sum_{s\in S} x_{is}=1 \qquad \forall\, i\in I
   \]

2. **Assignment only to opened centres**  
   \[
   x_{is}\le y_s \qquad \forall\, i\in I,\; s\in S
   \]

3. **Capacity of opened centres (at most 4 customers)**  
   \[
   \sum_{i\in I} x_{is}\le 4\,y_s \qquad \forall\, s\in S
   \]

All variables are binary.

---

**Data Mapping**

| Parameter | Source Table | Mapping Rule |
|-----------|--------------|--------------|
| \(f_s\) | `file_1_view_0` (columns: `Service Center`, `Fixed Opening Cost`) | For each row where `Service Center` = \(s\), set \(f_s =\) value of `Fixed Opening Cost`. |
| \(c_{is}\) | `file_0_view_0` (columns: `Customer`, `SC1`–`SC10`) | For each row where `Customer` = \(i\) and column = \(s\), set \(c_{is} =\) value in that cell. |

No numerical values are enumerated; the model references the data through the above mapping rules.