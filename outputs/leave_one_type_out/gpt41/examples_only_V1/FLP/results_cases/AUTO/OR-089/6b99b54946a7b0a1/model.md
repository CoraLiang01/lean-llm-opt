Let:
- I = {SC1, SC2, ..., SC10} be the set of candidate service centres (indexed by i)
- J = {C1, C2, ..., C15} be the set of customers (indexed by j)

Parameters:
- Fixed opening cost for each centre i ∈ I: 
  f = [385.1, 546.3, 485.2, 448.1, 324.1, 323.9, 296.5, 522.7, 448.7, 478.7]  
  where f_i is the fixed opening cost for centre SCi (i = 1 to 10, in order SC1–SC10)
- Service cost for assigning customer j ∈ J to centre i ∈ I: 
  c = [ [15.1, 21.2, 14.9, 18.8, 22.9, 16.8, 16.5, 9.4, 16.1, 17.3],   // C1
     [13.4, 16.3, 20.2, 19.6, 20.9, 22.1, 16.9, 9.4, 13.8, 11.7],     // C2
     [15.2, 18.8, 14.7, 21.7, 18.1, 18.6, 12.3, 11.2, 11.9, 20.4],   // C3
     [16.8, 19.1, 18.3, 18.8, 23.1, 15.7, 13.1, 8.6, 15.6, 22.2],    // C4
     [13.4, 18.6, 20.8, 19.8, 22.1, 18.1, 16.7, 12.1, 11.4, 18.2],   // C5
     [12.5, 22.5, 15.5, 14.9, 21.6, 21.3, 16.1, 10.7, 11.9, 14.6],   // C6
     [12.1, 17.1, 19.8, 18.6, 22.1, 20.7, 20.5, 12.2, 15.4, 18.7],   // C7
     [12.3, 15.7, 17.9, 21.3, 22.7, 15.3, 16.6, 11.4, 14.1, 20.1],   // C8
     [16.3, 21.3, 17.6, 20.8, 21.8, 17.2, 15.5, 12.6, 19.9, 19.1],   // C9
     [12.1, 18.7, 14.4, 20.1, 22.7, 14.1, 18.1, 11.4, 18.1, 17.4],   // C10
     [16.7, 18.7, 15.7, 19.9, 24.2, 18.7, 14.2, 13.1, 14.7, 16.1],   // C11
     [11.3, 23.8, 15.5, 17.3, 23.2, 17.7, 16.8, 14.5, 15.8, 17.8],   // C12
     [15.1, 20.5, 15.1, 18.4, 20.6, 17.9, 14.5, 8.5, 14.9, 13.9],    // C13
     [8.3, 20.7, 14.7, 20.4, 20.6, 14.8, 14.2, 11.5, 14.1, 15.1],    // C14
     [12.1, 16.3, 16.4, 15.1, 21.3, 19.1, 19.5, 16.7, 11.1, 18.7]    // C15
     ]
  where c[j][i] is the cost to serve customer Cj+1 from centre SCi+1 (indices 0-based).

Decision variables:
- y_i ∈ {0,1} for i ∈ I: y_i = 1 if centre i is opened, 0 otherwise
- x_{i,j} ∈ {0,1} for i ∈ I, j ∈ J: x_{i,j} = 1 if customer j is assigned to centre i, 0 otherwise

Mathematical Model:

Minimise:
\[
\text{Total Cost} = \sum_{i=1}^{10} f_i y_i + \sum_{i=1}^{10} \sum_{j=1}^{15} c_{j,i} x_{i,j}
\]
where:
- \(f_i\) is the fixed opening cost for centre SCi (see above vector)
- \(c_{j,i}\) is the cost to serve customer Cj from centre SCi (see above matrix)

Subject to:
1. Each customer is assigned to exactly one centre:
\[
\sum_{i=1}^{10} x_{i,j} = 1 \quad \forall j = 1,\ldots,15
\]

2. Customers can only be assigned to open centres:
\[
x_{i,j} \leq y_i \quad \forall i = 1,\ldots,10;\; j = 1,\ldots,15
\]

3. Each centre serves at most 4 customers:
\[
\sum_{j=1}^{15} x_{i,j} \leq 4 y_i \quad \forall i = 1,\ldots,10
\]

4. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i = 1,\ldots,10
\]
\[
x_{i,j} \in \{0,1\} \quad \forall i = 1,\ldots,10;\; j = 1,\ldots,15
\]

All parameters (fixed costs and service costs) are as retrieved above. This is a capacitated facility location problem with assignment and capacity constraints, using the exact costs from the provided CSV files.