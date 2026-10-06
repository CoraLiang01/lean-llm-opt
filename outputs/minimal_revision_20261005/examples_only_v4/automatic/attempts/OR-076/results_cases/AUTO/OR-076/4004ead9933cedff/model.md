Mathematical Model

Sets:
- \( W \): Set of warehouses, indexed by \( w \), from all "Warehouse ID" in file_1_view_0.
- \( C \): Set of customers, indexed by \( c \), from all "Customer ID" in file_2_view_0.

Parameters:
- \( f_w \): Fixed annual opening cost of warehouse \( w \), from "Fixed_Cost" in file_1_view_0.
- \( cap_w \): Maximum service capacity of warehouse \( w \), from "Capacity" in file_1_view_0.
- \( d_c \): Demand of customer \( c \), from "Demand" in file_2_view_0.
- \( t_{w,c} \): Transportation cost per unit from warehouse \( w \) to customer \( c \), from file_0_view_0, with row "Warehouse ID" and column "Customer ID".

Decision Variables:
- \( y_w \in \{0,1\} \): 1 if warehouse \( w \) is opened, 0 otherwise.
- \( x_{w,c} \geq 0 \): Amount of customer \( c \)'s demand supplied from warehouse \( w \).

Objective:
\[
\min \sum_{w \in W} f_w y_w + \sum_{w \in W} \sum_{c \in C} t_{w,c} x_{w,c}
\]

Subject to:
1. Demand satisfaction for each customer:
\[
\sum_{w \in W} x_{w,c} = d_c \quad \forall c \in C
\]

2. Warehouse capacity:
\[
\sum_{c \in C} x_{w,c} \leq cap_w y_w \quad \forall w \in W
\]

3. Variable domains:
\[
y_w \in \{0,1\} \quad \forall w \in W
\]
\[
x_{w,c} \geq 0 \quad \forall w \in W,\, c \in C
\]

Data Mapping

- \( W \): All "Warehouse ID" in table_id file_1_view_0
- \( C \): All "Customer ID" in table_id file_2_view_0
- \( f_w \): "Fixed_Cost" in table_id file_1_view_0, indexed by "Warehouse ID"
- \( cap_w \): "Capacity" in table_id file_1_view_0, indexed by "Warehouse ID"
- \( d_c \): "Demand" in table_id file_2_view_0, indexed by "Customer ID"
- \( t_{w,c} \): file_0_view_0, row "Warehouse ID", column "Customer ID" (matrix: warehouse-customer transportation cost)