##### Abstract Model

**Index Set**  
\(I\) : set of vehicle types.

**Parameters**  
\(b_i\) : benefit coefficient for vehicle type \(i\).  
\(c_i\) : daily inventory limit for vehicle type \(i\).  
\(C\)   : total inventory capacity per day, defined as \(C=\sum_{i\in I} c_i\).

**Decision Variables**  
\(x_i\) : number of vehicles of type \(i\) to order per day, integer, \(x_i \ge 0\).

**Objective Function**  
\[
\quad \quad \max \quad \sum_{i\in I} b_i \cdot x_i
\]

**Constraints**

1. **Daily inventory limits**  
\[
\quad \quad x_i \le c_i, \quad \forall i \in I
\]

2. **Total inventory capacity**  
\[
\sum_{i\in I} x_i \le C
\]

3. **Variable domain**  
\[
x_i \in \mathbb{Z}, \quad x_i \ge 0, \quad \forall i \in I
\]

**Data Mapping**

- Parameter \(b_i\) : table_id **file_1_view_0**, column **Value**.  
- Parameter \(c_i\) : table_id **file_0_view_0**, column **Capacity**.  
- Parameter \(C\)   : derived from table_id **file_0_view_0** by summing all **Capacity** values.