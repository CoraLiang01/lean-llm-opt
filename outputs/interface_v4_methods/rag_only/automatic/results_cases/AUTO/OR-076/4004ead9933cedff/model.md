Let:
- I = set of warehouses = {W1, W2, W3, W4, W5, W6, W7, W8, W9, W10}
- J = set of customers = {C1, C2, ..., C20}

Parameters:
- Fixed opening cost for warehouse i: 
  - f = [2000, 2500, 1800, 3200, 1500, 4000, 2800, 1950, 3500, 2200] 
    (where f[i] corresponds to Wi, i=1..10)
- Capacity of warehouse i:
  - cap = [1000, 1500, 1200, 2000, 800, 2500, 1800, 1100, 2100, 1300]
    (where cap[i] corresponds to Wi, i=1..10)
- Demand of customer j:
  - d = [800, 600, 500, 700, 450, 950, 350, 850, 400, 750, 900, 550, 650, 820, 480, 920, 320, 780, 520, 680]
    (where d[j] corresponds to Cj, j=1..20)
- Transportation cost from warehouse i to customer j:
  - c = 10x20 matrix, where c[i][j] is the cost from Wi to Cj:

|      | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C15 | C16 | C17 | C18 | C19 | C20 |
|------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| W1   | 10 | 15 | 20 | 11 | 16 | 18 | 7  | 12 | 22 | 9   | 14  | 19  | 25  | 13  | 17  | 6   | 21  | 15  | 8   | 10  |
| W2   | 18 | 12 | 9  | 14 | 10 | 5  | 19 | 23 | 11 | 16  | 20  | 8   | 15  | 22  | 7   | 13  | 24  | 17  | 12  | 6   |
| W3   | 13 | 17 | 15 | 8  | 12 | 21 | 16 | 10 | 5  | 24  | 13  | 22  | 7   | 19  | 14  | 18  | 9   | 25  | 11  | 16  |
| W4   | 7  | 22 | 11 | 16 | 20 | 8  | 15 | 19 | 13 | 25  | 6   | 14  | 21  | 9   | 23  | 17  | 10  | 18  | 24  | 5   |
| W5   | 16 | 9  | 25 | 13 | 7  | 10 | 23 | 14 | 18 | 21  | 5   | 17  | 9   | 24  | 12  | 20  | 6   | 15  | 19  | 11  |
| W6   | 22 | 6  | 14 | 19 | 23 | 11 | 8  | 17 | 9  | 12  | 15  | 24  | 5   | 20  | 10  | 25  | 13  | 7   | 18  | 16  |
| W7   | 8  | 25 | 17 | 9  | 14 | 22 | 11 | 6  | 16 | 20  | 18  | 13  | 24  | 5   | 19  | 12  | 23  | 10  | 7   | 15  |
| W8   | 19 | 11 | 7  | 21 | 15 | 24 | 13 | 16 | 20 | 8   | 17  | 10  | 12  | 23  | 5   | 14  | 22  | 9   | 16  | 25  |
| W9   | 12 | 20 | 5  | 23 | 17 | 14 | 9  | 25 | 18 | 11  | 16  | 21  | 10  | 7   | 24  | 15  | 19  | 6   | 13  | 22  |
| W10  | 25 | 14 | 22 | 5  | 19 | 12 | 24 | 7  | 15 | 17  | 23  | 6   | 16  | 10  | 20  | 9   | 18  | 11  | 25  | 14  |

Decision variables:
- y_i ∈ {0,1} for i ∈ I: 1 if warehouse i is opened, 0 otherwise
- x_{ij} ≥ 0 for i ∈ I, j ∈ J: amount of customer j's demand served from warehouse i

Mathematical Model:

Minimize total cost:
\[
\text{Minimize} \quad Z = \sum_{i=1}^{10} f_i y_i + \sum_{i=1}^{10} \sum_{j=1}^{20} c_{ij} x_{ij}
\]
where:
- \( f = [2000, 2500, 1800, 3200, 1500, 4000, 2800, 1950, 3500, 2200] \)
- \( c_{ij} \) as given in the table above

Subject to:
1. Demand fulfillment for each customer:
\[
\sum_{i=1}^{10} x_{ij} = d_j \quad \forall j=1,\ldots,20
\]
where \( d = [800, 600, 500, 700, 450, 950, 350, 850, 400, 750, 900, 550, 650, 820, 480, 920, 320, 780, 520, 680] \)

2. Warehouse capacity:
\[
\sum_{j=1}^{20} x_{ij} \leq cap_i \cdot y_i \quad \forall i=1,\ldots,10
\]
where \( cap = [1000, 1500, 1200, 2000, 800, 2500, 1800, 1100, 2100, 1300] \)

3. Non-negativity and binary constraints:
\[
x_{ij} \geq 0 \quad \forall i=1,\ldots,10; \; j=1,\ldots,20
\]
\[
y_i \in \{0,1\} \quad \forall i=1,\ldots,10
\]

Summary:
- All parameters (fixed costs, capacities, demands, transportation costs) are explicitly listed above.
- The objective is to minimize the sum of fixed opening costs and transportation costs.
- All customer demand must be met.
- No warehouse can serve more than its capacity, and only if it is open.
- Decision variables are warehouse open/close (binary) and customer allocation (continuous, non-negative).