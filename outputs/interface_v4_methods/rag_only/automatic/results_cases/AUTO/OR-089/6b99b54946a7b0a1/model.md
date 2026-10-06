Let us define the following mathematical model for the service centre location and customer assignment problem, using the data retrieved from service_centers_fixed_costs.csv and expanded_customer_service_costs.csv.

Sets:
- Let I = {SC1, SC2, ..., SC10} be the set of candidate service centres, indexed by i.
- Let J = {C1, C2, ..., C15} be the set of customers, indexed by j.

Parameters:
- Fixed opening cost for each centre i ∈ I:
    - f_i:
        - f_SC1 = 385.1
        - f_SC2 = 546.3
        - f_SC3 = 485.2
        - f_SC4 = 448.1
        - f_SC5 = 324.1
        - f_SC6 = 323.9
        - f_SC7 = 296.5
        - f_SC8 = 522.7
        - f_SC9 = 448.7
        - f_SC10 = 478.7
- Service cost for assigning customer j ∈ J to centre i ∈ I:
    - c_{ij} (matrix below):

|      | SC1  | SC2  | SC3  | SC4  | SC5  | SC6  | SC7  | SC8  | SC9  | SC10 |
|------|------|------|------|------|------|------|------|------|------|-------|
| C1   | 15.1 | 21.2 | 14.9 | 18.8 | 22.9 | 16.8 | 16.5 | 9.4  | 16.1 | 17.3  |
| C2   | 13.4 | 16.3 | 20.2 | 19.6 | 20.9 | 22.1 | 16.9 | 9.4  | 13.8 | 11.7  |
| C3   | 15.2 | 18.8 | 14.7 | 21.7 | 18.1 | 18.6 | 12.3 | 11.2 | 11.9 | 20.4  |
| C4   | 16.8 | 19.1 | 18.3 | 18.8 | 23.1 | 15.7 | 13.1 | 8.6  | 15.6 | 22.2  |
| C5   | 13.4 | 18.6 | 20.8 | 19.8 | 22.1 | 18.1 | 16.7 | 12.1 | 11.4 | 18.2  |
| C6   | 12.5 | 22.5 | 15.5 | 14.9 | 21.6 | 21.3 | 16.1 | 10.7 | 11.9 | 14.6  |
| C7   | 12.1 | 17.1 | 19.8 | 18.6 | 22.1 | 20.7 | 20.5 | 12.2 | 15.4 | 18.7  |
| C8   | 12.3 | 15.7 | 17.9 | 21.3 | 22.7 | 15.3 | 16.6 | 11.4 | 14.1 | 20.1  |
| C9   | 16.3 | 21.3 | 17.6 | 20.8 | 21.8 | 17.2 | 15.5 | 12.6 | 19.9 | 19.1  |
| C10  | 12.1 | 18.7 | 14.4 | 20.1 | 22.7 | 14.1 | 18.1 | 11.4 | 18.1 | 17.4  |
| C11  | 16.7 | 18.7 | 15.7 | 19.9 | 24.2 | 18.7 | 14.2 | 13.1 | 14.7 | 16.1  |
| C12  | 11.3 | 23.8 | 15.5 | 17.3 | 23.2 | 17.7 | 16.8 | 14.5 | 15.8 | 17.8  |
| C13  | 15.1 | 20.5 | 15.1 | 18.4 | 20.6 | 17.9 | 14.5 | 8.5  | 14.9 | 13.9  |
| C14  | 8.3  | 20.7 | 14.7 | 20.4 | 20.6 | 14.8 | 14.2 | 11.5 | 14.1 | 15.1  |
| C15  | 12.1 | 16.3 | 16.4 | 15.1 | 21.3 | 19.1 | 19.5 | 16.7 | 11.1 | 18.7  |

Decision Variables:
- y_i ∈ {0,1} for each i ∈ I: y_i = 1 if centre i is opened, 0 otherwise.
- x_{ij} ∈ {0,1} for each i ∈ I, j ∈ J: x_{ij} = 1 if customer j is assigned to centre i, 0 otherwise.

Mathematical Model:

Objective:
Minimise total cost:
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where f_i and c_{ij} are as given above.

Subject to:

1. Each customer is assigned to exactly one centre:
\[
\sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
\]

2. Customers can only be assigned to open centres:
\[
x_{ij} \leq y_i \quad \forall i \in I, \forall j \in J
\]

3. Each centre serves at most 4 customers:
\[
\sum_{j \in J} x_{ij} \leq 4 y_i \quad \forall i \in I
\]

4. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in I
\]
\[
x_{ij} \in \{0,1\} \quad \forall i \in I, \forall j \in J
\]

All parameters (fixed costs and service costs) are as listed above, with full vectors and matrices preserved from the CSV data. This model minimises the sum of fixed opening costs and customer–centre service costs, ensuring every customer is assigned to exactly one open centre, and no centre serves more than 4 customers.