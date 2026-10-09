Mathematical Model (Uncapacitated Facility Location Problem with Capacity Constraint):

Sets:
- Let S = {SC1, SC2, ..., SC10} be the set of candidate service centres, indexed by s.
- Let C = {C1, C2, ..., C15} be the set of customers, indexed by c.

Parameters:
- f_s: Fixed opening cost for centre s.  
  Data Mapping: file_1_view_0, columns: Service Center, Fixed Opening Cost; mapping s ↔ Service Center, f_s = Fixed Opening Cost.
- c_{c,s}: Cost to serve customer c from centre s.  
  Data Mapping: file_0_view_0, columns: Customer, SC1–SC10; mapping (c,s) ↔ (Customer, SC*), c_{c,s} = value at (Customer=c, SC*=s).

Decision Variables:
- y_s ∈ {0,1}: 1 if centre s is opened, 0 otherwise.
- x_{c,s} ∈ {0,1}: 1 if customer c is assigned to centre s, 0 otherwise.

Objective:
Minimise total cost:
\[
\min \sum_{s \in S} f_s y_s + \sum_{c \in C} \sum_{s \in S} c_{c,s} x_{c,s}
\]

Subject to:
1. Each customer is assigned to exactly one centre:
\[
\forall c \in C: \quad \sum_{s \in S} x_{c,s} = 1
\]

2. Customers can only be assigned to open centres:
\[
\forall c \in C, \forall s \in S: \quad x_{c,s} \leq y_s
\]

3. Each centre serves at most 4 customers:
\[
\forall s \in S: \quad \sum_{c \in C} x_{c,s} \leq 4
\]

4. Variable domains:
\[
y_s \in \{0,1\} \quad \forall s \in S
\]
\[
x_{c,s} \in \{0,1\} \quad \forall c \in C, s \in S
\]

Data Mapping:
- S = all Service Center values in file_1_view_0["Service Center"]
- C = all Customer values in file_0_view_0["Customer"]
- f_s: file_1_view_0, columns: Service Center, Fixed Opening Cost; s ↔ Service Center, f_s = Fixed Opening Cost
- c_{c,s}: file_0_view_0, columns: Customer, SC1–SC10; (c,s) ↔ (Customer, SC*), c_{c,s} = value at (Customer=c, SC*=s)

All parameters and index sets are bound exactly to the current CSV data.