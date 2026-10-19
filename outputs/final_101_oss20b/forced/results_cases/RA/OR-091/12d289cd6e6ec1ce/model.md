**Retrieved Operations Research Courses**

| course_id | course_name                                 | credits | interest_points |
|-----------|---------------------------------------------|---------|-----------------|
| C22       | Operations Research: Linear Programming     | 5       | 95              |
| C23       | Integer Programming                        | 5       | 92              |
| C24       | Stochastic Processes                        | 4       | 86              |
| C25       | Simulation Modeling                         | 4       | 82              |
| C26       | Network Flows                               | 4       | 85              |
| C27       | Queueing Theory                             | 4       | 80              |
| C28       | Revenue Management                          | 4       | 88              |

---

### Mathematical Model

Let  
\(y_i \in \{0,1\}\) be a binary decision variable that equals 1 if course \(i\) is selected, 0 otherwise.  
Index \(i\) runs over the seven Operations Research courses listed above.

#### Objective Function  
\[
\max \sum_{i} \text{interest_points}_i \; y_i
\]

#### Constraints  

1. **Credit limit (total credits ≤ 20)**  
\[
\sum_{i} \text{credits}_i \; y_i \;\leq\; 20
\]

2. **Binary decision variables**  
\[
y_i \in \{0,1\}, \quad \forall i
\]

---

### Explicit Formulation with Data

\[
\max \; 95\,y_{\text{C22}} + 92\,y_{\text{C23}} + 86\,y_{\text{C24}} + 82\,y_{\text{C25}} + 85\,y_{\text{C26}} + 80\,y_{\text{C27}} + 88\,y_{\text{C28}}
\]

\[
\text{s.t.}\;\; 5\,y_{\text{C22}} + 5\,y_{\text{C23}} + 4\,y_{\text{C24}} + 4\,y_{\text{C25}} + 4\,y_{\text{C26}} + 4\,y_{\text{C27}} + 4\,y_{\text{C28}} \;\leq\; 20
\]

\[
y_{\text{C22}},\,y_{\text{C23}},\,y_{\text{C24}},\,y_{\text{C25}},\,y_{\text{C26}},\,y_{\text{C27}},\,y_{\text{C28}} \;\in\; \{0,1\}
\]

This model selects the optimal subset of Operations Research courses that maximizes total interest points while keeping the total credit load within 20 credits.