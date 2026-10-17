**Mathematical Model**

Let  

* \(I=\{C1,\dots ,C15\}\) – set of customers  
* \(S=\{SC1,\dots ,SC10\}\) – set of candidate service centres  

Parameters  

* \(f_s\) – fixed opening cost of centre \(s\)  
* \(c_{i,s}\) – cost of serving customer \(i\) from centre \(s\)  

Decision variables  

* \(y_s \in \{0,1\}\) – 1 if centre \(s\) is opened, 0 otherwise  
* \(x_{i,s} \in \{0,1\}\) – 1 if customer \(i\) is assigned to centre \(s\), 0 otherwise  

**Objective**

\[
\min \;\sum_{s\in S} f_s\,y_s \;+\; \sum_{i\in I}\sum_{s\in S} c_{i,s}\,x_{i,s}
\]

**Constraints**

1. **Customer assignment**

\[
\sum_{s\in S} x_{i,s}=1 \qquad \forall i\in I
\]

2. **Assignment only to opened centres**

\[
x_{i,s}\le y_s \qquad \forall i\in I,\; \forall s\in S
\]

3. **Capacity of opened centres**

\[
\sum_{i\in I} x_{i,s}\le 4\,y_s \qquad \forall s\in S
\]

4. **Variable domains**

\[
x_{i,s}\in\{0,1\},\; y_s\in\{0,1\}\qquad \forall i\in I,\; \forall s\in S
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column | Mapping Rule |
|-----------|--------------|------------|--------------|--------------|
| \(f_s\) | `file_1_view_0` | `Service Center` | `Fixed Opening Cost` | \(f_s =\) value where `Service Center` = \(s\) |
| \(c_{i,s}\) | `file_0_view_0` | `Customer` | `SC1`–`SC10` | \(c_{i,s} =\) value where `Customer` = \(i\) and column = \(s\) |

All indices \(i\) and \(s\) are taken from the full sets of customers and service centres present in the respective tables. No additional enumeration of numerical values is provided.