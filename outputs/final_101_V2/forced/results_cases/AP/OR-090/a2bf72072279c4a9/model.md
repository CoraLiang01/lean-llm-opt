##### Sets and Indices

- Let $I = \{1, 2, \ldots, 100\}$ index the products, with product $i$ corresponding to $P_i$.
- Let $K = \{1,2,3\}$ index the resources, with $k=1$ for R1, $k=2$ for R2, $k=3$ for R3.

##### Parameters

For each product $i$ ($i=1,\ldots,100$):

- $profit\_per\_unit_i$: profit per unit of product $i$ (from data)
- $r1\_per\_unit_i$: units of resource R1 consumed per unit of product $i$
- $r2\_per\_unit_i$: units of resource R2 consumed per unit of product $i$
- $r3\_per\_unit_i$: units of resource R3 consumed per unit of product $i$
- $upper\_demand\_units_i$: maximum market demand (units) for product $i$
- $batch\_size\_units = 10$

Resource capacities:

- $capacity_1 = 27380.54$ (R1)
- $capacity_2 = 22245.11$ (R2)
- $capacity_3 = 15147.73$ (R3)

##### Decision Variables

- $x_i \in \mathbb{Z}_+$: number of batches of product $i$ to produce (integer, $\geq 0$)

##### Objective Function

$$
\max \sum_{i=1}^{100} \left( batch\_size\_units \cdot x_i \cdot profit\_per\_unit_i \right)
$$

##### Constraints

###### 1. Resource Capacity Constraints

For each resource $k$:

- For R1 ($k=1$):
  $$
  \sum_{i=1}^{100} batch\_size\_units \cdot x_i \cdot r1\_per\_unit_i \leq capacity_1
  $$
- For R2 ($k=2$):
  $$
  \sum_{i=1}^{100} batch\_size\_units \cdot x_i \cdot r2\_per\_unit_i \leq capacity_2
  $$
- For R3 ($k=3$):
  $$
  \sum_{i=1}^{100} batch\_size\_units \cdot x_i \cdot r3\_per\_unit_i \leq capacity_3
  $$

###### 2. Demand Upper Bound Constraints

For each product $i$:
$$
batch\_size\_units \cdot x_i \leq upper\_demand\_units_i
$$

###### 3. Integer and Nonnegativity Constraints

For each product $i$:
$$
x_i \in \mathbb{Z}_+, \quad x_i \geq 0
$$

##### Retrieved Information

```json
{
  "products": [
    {"product": "P1", "profit_per_unit": 6.7, "r1_per_unit": 2.37, "r2_per_unit": 0.61, "r3_per_unit": 2.03, "upper_demand_units": 317, "batch_size_units": 10},
    {"product": "P2", "profit_per_unit": 10.96, "r1_per_unit": 4.79, "r2_per_unit": 2.73, "r3_per_unit": 0.53, "upper_demand_units": 106, "batch_size_units": 10},
    {"product": "P3", "profit_per_unit": 9.4, "r1_per_unit": 3.87, "r2_per_unit": 1.6, "r3_per_unit": 0.74, "upper_demand_units": 386, "batch_size_units": 10},
    {"product": "P4", "profit_per_unit": 11.13, "r1_per_unit": 3.31, "r2_per_unit": 2.28, "r3_per_unit": 2.73, "upper_demand_units": 441, "batch_size_units": 10},
    {"product": "P5", "profit_per_unit": 10.29, "r1_per_unit": 1.46, "r2_per_unit": 3.68, "r3_per_unit": 1.94, "upper_demand_units": 63, "batch_size_units": 10},
    {"product": "P6", "profit_per_unit": 8.06, "r1_per_unit": 1.46, "r2_per_unit": 1.37, "r3_per_unit": 0.32, "upper_demand_units": 221, "batch_size_units": 10},
    {"product": "P7", "profit_per_unit": 6.94, "r1_per_unit": 1.04, "r2_per_unit": 1.94, "r3_per_unit": 0.57, "upper_demand_units": 441, "batch_size_units": 10},
    {"product": "P8", "profit_per_unit": 11.44, "r1_per_unit": 4.44, "r2_per_unit": 3.14, "r3_per_unit": 2.09, "upper_demand_units": 489, "batch_size_units": 10},
    {"product": "P9", "profit_per_unit": 9.13, "r1_per_unit": 3.32, "r2_per_unit": 1.3, "r3_per_unit": 0.31, "upper_demand_units": 277, "batch_size_units": 10},
    {"product": "P10", "profit_per_unit": 7.83, "r1_per_unit": 3.77, "r2_per_unit": 0.77, "r3_per_unit": 0.73, "upper_demand_units": 121, "batch_size_units": 10},
    {"product": "P11", "profit_per_unit": 7.07, "r1_per_unit": 0.89, "r2_per_unit": 1.51, "r3_per_unit": 1.78, "upper_demand_units": 309, "batch_size_units": 10},
    {"product": "P12", "profit_per_unit": 9.49, "r1_per_unit": 4.87, "r2_per_unit": 1.06, "r3_per_unit": 2.17, "upper_demand_units": 499, "batch_size_units": 10},
    {"product": "P13", "profit_per_unit": 10.89, "r1_per_unit": 4.3, "r2_per_unit": 3.75, "r3_per_unit": 2.06, "upper_demand_units": 428, "batch_size_units": 10},
    {"product": "P14", "profit_per_unit": 10.21, "r1_per_unit": 1.69, "r2_per_unit": 3.33, "r3_per_unit": 0.91, "upper_demand_units": 608, "batch_size_units": 10},
    {"product": "P15", "profit_per_unit": 10.14, "r1_per_unit": 1.56, "r2_per_unit": 2.72, "r3_per_unit": 2.22, "upper_demand_units": 551, "batch_size_units": 10},
    {"product": "P16", "profit_per_unit": 9.5, "r1_per_unit": 1.57, "r2_per_unit": 3.55, "r3_per_unit": 0.94, "upper_demand_units": 411, "batch_size_units": 10},
    {"product": "P17", "profit_per_unit": 9.07, "r1_per_unit": 2.08, "r2_per_unit": 3.31, "r3_per_unit": 1.18, "upper_demand_units": 188, "batch_size_units": 10},
    {"product": "P18", "profit_per_unit": 8.26, "r1_per_unit": 3.0, "r2_per_unit": 1.15, "r3_per_unit": 2.32, "upper_demand_units": 113, "batch_size_units": 10},
    {"product": "P19", "profit_per_unit": 9.65, "r1_per_unit": 2.61, "r2_per_unit": 3.62, "r3_per_unit": 2.05, "upper_demand_units": 143, "batch_size_units": 10},
    {"product": "P20", "profit_per_unit": 8.79, "r1_per_unit": 2.02, "r2_per_unit": 2.39, "r3_per_unit": 2.59, "upper_demand_units": 441, "batch_size_units": 10},
    {"product": "P21", "profit_per_unit": 11.3, "r1_per_unit": 3.37, "r2_per_unit": 3.33, "r3_per_unit": 2.08, "upper_demand_units": 69, "batch_size_units": 10},
    {"product": "P22", "profit_per_unit": 10.09, "r1_per_unit": 1.39, "r2_per_unit": 3.64, "r3_per_unit": 1.83, "upper_demand_units": 513, "batch_size_units": 10},
    {"product": "P23", "profit_per_unit": 7.98, "r1_per_unit": 2.03, "r2_per_unit": 1.61, "r3_per_unit": 0.55, "upper_demand_units": 528, "batch_size_units": 10},
    {"product": "P24", "profit_per_unit": 7.06, "r1_per_unit": 2.34, "r2_per_unit": 0.89, "r3_per_unit": 1.29, "upper_demand_units": 609, "batch_size_units": 10},
    {"product": "P25", "profit_per_unit": 9.57, "r1_per_unit": 2.72, "r2_per_unit": 1.3, "r3_per_unit": 1.02, "upper_demand_units": 195, "batch_size_units": 10},
    {"product": "P26", "profit_per_unit": 10.67, "r1_per_unit": 4.1, "r2_per_unit": 1.99, "r3_per_unit": 0.96, "upper_demand_units": 188, "batch_size_units": 10},
    {"product": "P27", "profit_per_unit": 10.38, "r1_per_unit": 1.64, "r2_per_unit": 3.36, "r3_per_unit": 2.93, "upper_demand_units": 613, "batch_size_units": 10},
    {"product": "P28", "profit_per_unit": 10.76, "r1_per_unit": 2.96, "r2_per_unit": 3.51, "r3_per_unit": 1.36, "upper_demand_units": 83, "batch_size_units": 10},
    {"product": "P29", "profit_per_unit": 9.03, "r1_per_unit": 3.29, "r2_per_unit": 0.52, "r3_per_unit": 2.71, "upper_demand_units": 153, "batch_size_units": 10},
    {"product": "P30", "profit_per_unit": 7.38, "r1_per_unit": 1.0, "r2_per_unit": 2.29, "r3_per_unit": 2.0, "upper_demand_units": 86, "batch_size_units": 10},
    {"product": "P31", "profit_per_unit": 9.87, "r1_per_unit": 3.35, "r2_per_unit": 1.96, "r3_per_unit": 2.45, "upper_demand_units": 231, "batch_size_units": 10},
    {"product": "P32", "profit_per_unit": 8.33, "r1_per_unit": 1.52, "r2_per_unit": 1.28, "r3_per_unit": 1.66, "upper_demand_units": 442, "batch_size_units": 10},
    {"product": "P33", "profit_per_unit": 5.54, "r1_per_unit": 1.07, "r2_per_unit": 0.92, "r3_per_unit": 1.86, "upper_demand_units": 267, "batch_size_units": 10},
    {"product": "P34", "profit_per_unit": 9.64, "r1_per_unit": 4.79, "r2_per_unit": 1.68, "r3_per_unit": 1.63, "upper_demand_units": 177, "batch_size_units": 10},
    {"product": "P35", "profit_per_unit": 10.63, "r1_per_unit": 4.86, "r2_per_unit": 3.8, "r3_per_unit": 0.83, "upper_demand_units": 207, "batch_size_units": 10},
    {"product": "P36", "profit_per_unit": 9.28, "r1_per_unit": 4.2, "r2_per_unit": 1.63, "r3_per_unit": 2.25, "upper_demand_units": 259, "batch_size_units": 10},
    {"product": "P37", "profit_per_unit": 9.54, "r1_per_unit": 2.08, "r2_per_unit": 2.32, "r3_per_unit": 1.06, "upper_demand_units": 279, "batch_size_units": 10},
    {"product": "P38", "profit_per_unit": 8.33, "r1_per_unit": 1.21, "r2_per_unit": 2.96, "r3_per_unit": 0.37, "upper_demand_units": 53, "batch_size_units": 10},
    {"product": "P39", "profit_per_unit": 10.07, "r1_per_unit": 3.67, "r2_per_unit": 1.77, "r3_per_unit": 2.04, "upper_demand_units": 118, "batch_size_units": 10},
    {"product": "P40", "profit_per_unit": 8.84, "r1_per_unit": 2.65, "r2_per_unit": 3.9, "r3_per_unit": 0.78, "upper_demand_units": 536, "batch_size_units": 10},
    {"product": "P41", "profit_per_unit": 10.24, "r1_per_unit": 1.31, "r2_per_unit": 3.87, "r3_per_unit": 2.84, "upper_demand_units": 297, "batch_size_units": 10},
    {"product": "P42", "profit_per_unit": 9.63, "r1_per_unit": 2.88, "r2_per_unit": 1.38, "r3_per_unit": 2.88, "upper_demand_units": 274, "batch_size_units": 10},
    {"product": "P43", "profit_per_unit": 7.6, "r1_per_unit": 0.94, "r2_per_unit": 2.24, "r3_per_unit": 2.77, "upper_demand_units": 563, "batch_size_units": 10},
    {"product": "P44", "profit_per_unit": 10.11, "r1_per_unit": 4.62, "r2_per_unit": 1.55, "r3_per_unit": 1.3, "upper_demand_units": 218, "batch_size_units": 10},
    {"product": "P45", "profit_per_unit": 6.83, "r1_per_unit": 1.89, "r2_per_unit": 1.5, "r3_per_unit": 0.34, "upper_demand_units": 212, "batch_size_units": 10},
    {"product": "P46", "profit_per_unit": 10.04, "r1_per_unit": 3.58, "r2_per_unit": 0.63, "r3_per_unit": 2.81, "upper_demand_units": 214, "batch_size_units": 10},
    {"product": "P47", "profit_per_unit": 9.43, "r1_per_unit": 2.11, "r2_per_unit": 2.63, "r3_per_unit": 1.46, "upper_demand_units": 437, "batch_size_units": 10},
    {"product": "P48", "profit_per_unit": 9.16, "r1_per_unit": 2.98, "r2_per_unit": 2.26, "r3_per_unit": 2.91, "upper_demand_units": 238, "batch_size_units": 10},
    {"product": "P49", "profit_per_unit": 8.99, "r1_per_unit": 3.1, "r2_per_unit": 0.68, "r3_per_unit": 2.9, "upper_demand_units": 561, "batch_size_units": 10},
    {"product": "P50", "profit_per_unit": 8.8, "r1_per_unit": 1.58, "r2_per_unit": 1.48, "r3_per_unit": 2.6, "upper_demand_units": 73, "batch_size_units": 10},
    {"product": "P51", "profit_per_unit": 12.11, "r1_per_unit": 4.87, "r2_per_unit": 3.68, "r3_per_unit": 1.1, "upper_demand_units": 209, "batch_size_units": 10},
    {"product": "P52", "profit_per_unit": 10.73, "r1_per_unit": 4.06, "r2_per_unit": 1.34, "r3_per_unit": 1.34, "upper_demand_units": 466, "batch_size_units": 10},
    {"product": "P53", "profit_per_unit": 11.38, "r1_per_unit": 4.75, "r2_per_unit": 1.01, "r3_per_unit": 2.6, "upper_demand_units": 383, "batch_size_units": 10},
    {"product": "P54", "profit_per_unit": 9.67, "r1_per_unit": 4.56, "r2_per_unit": 2.21, "r3_per_unit": 1.16, "upper_demand_units": 541, "batch_size_units": 10},
    {"product": "P55", "profit_per_unit": 9.38, "r1_per_unit": 3.31, "r2_per_unit": 3.95, "r3_per_unit": 0.76, "upper_demand_units": 409, "batch_size_units": 10},
    {"product": "P56", "profit_per_unit": 10.97, "r1_per_unit": 4.67, "r2_per_unit": 1.35, "r3_per_unit": 1.8, "upper_demand_units": 386, "batch_size_units": 10},
    {"product": "P57", "profit_per_unit": 7.89, "r1_per_unit": 1.17, "r2_per_unit": 2.85, "r3_per_unit": 2.83, "upper_demand_units": 544, "batch_size_units": 10},
    {"product": "P58", "profit_per_unit": 9.78, "r1_per_unit": 1.62, "r2_per_unit": 3.17, "r3_per_unit": 2.18, "upper_demand_units": 555, "batch_size_units": 10},
    {"product": "P59", "profit_per_unit": 8.56, "r1_per_unit": 0.99, "r2_per_unit": 1.33, "r3_per_unit": 1.84, "upper_demand_units": 371, "batch_size_units": 10},
    {"product": "P60", "profit_per_unit": 9.01, "r1_per_unit": 2.17, "r2_per_unit": 3.05, "r3_per_unit": 0.56, "upper_demand_units": 261, "batch_size_units": 10},
    {"product": "P61", "profit_per_unit": 8.66, "r1_per_unit": 2.43, "r2_per_unit": 1.79, "r3_per_unit": 1.96, "upper_demand_units": 345, "batch_size_units": 10},
    {"product": "P62", "profit_per_unit": 10.42, "r1_per_unit": 1.94, "r2_per_unit": 2.71, "r3_per_unit": 2.97, "upper_demand_units": 176, "batch_size_units": 10},
    {"product": "P63", "profit_per_unit": 10.38, "r1_per_unit": 4.28, "r2_per_unit": 2.72, "r3_per_unit": 0.68, "upper_demand_units": 114, "batch_size_units": 10},
    {"product": "P64", "profit_per_unit": 9.3, "r1_per_unit": 2.3, "r2_per_unit": 2.38, "r3_per_unit": 1.7, "upper_demand_units": 296, "batch_size_units": 10},
    {"product": "P65", "profit_per_unit": 9.63, "r1_per_unit": 1.98, "r2_per_unit": 0.82, "r3_per_unit": 2.67, "upper_demand_units": 459, "batch_size_units": 10},
    {"product": "P66", "profit_per_unit": 10.78, "r1_per_unit": 3.08, "r2_per_unit": 3.42, "r3_per_unit": 2.3, "upper_demand_units": 489, "batch_size_units": 10},
    {"product": "P67", "profit_per_unit": 9.46, "r1_per_unit": 1.39, "r2_per_unit": 1.62, "r3_per_unit": 2.18, "upper_demand_units": 226, "batch_size_units": 10},
    {"product": "P68", "profit_per_unit": 11.43, "r1_per_unit": 4.17, "r2_per_unit": 1.15, "r3_per_unit": 2.2, "upper_demand_units": 211, "batch_size_units": 10},
    {"product": "P69", "profit_per_unit": 5.49, "r1_per_unit": 1.11, "r2_per_unit": 0.64, "r3_per_unit": 1.27, "upper_demand_units": 135, "batch_size_units": 10},
    {"product": "P70", "profit_per_unit": 9.97, "r1_per_unit": 4.94, "r2_per_unit": 2.57, "r3_per_unit": 1.09, "upper_demand_units": 575, "batch_size_units": 10},
    {"product": "P71", "profit_per_unit": 10.48, "r1_per_unit": 4.04, "r2_per_unit": 2.87, "r3_per_unit": 2.49, "upper_demand_units": 83, "batch_size_units": 10},
    {"product": "P72", "profit_per_unit": 6.21, "r1_per_unit": 1.63, "r2_per_unit": 0.56, "r3_per_unit": 2.49, "upper_demand_units": 171, "batch_size_units": 10},
    {"product": "P73", "profit_per_unit": 7.2, "r1_per_unit": 0.82, "r2_per_unit": 2.29, "r3_per_unit": 2.64, "upper_demand_units": 495, "batch_size_units": 10},
    {"product": "P74", "profit_per_unit": 11.32, "r1_per_unit": 4.22, "r2_per_unit": 1.29, "r3_per_unit": 2.77, "upper_demand_units": 217, "batch_size_units": 10},
    {"product": "P75", "profit_per_unit": 9.49, "r1_per_unit": 3.77, "r2_per_unit": 2.76, "r3_per_unit": 1.68, "upper_demand_units": 264, "batch_size_units": 10},
    {"product": "P76", "profit_per_unit": 8.97, "r1_per_unit": 3.86, "r2_per_unit": 1.11, "r3_per_unit": 1.65, "upper_demand_units": 102, "batch_size_units": 10},
    {"product": "P77", "profit_per_unit": 12.73, "r1_per_unit": 4.04, "r2_per_unit": 2.92, "r3_per_unit": 2.46, "upper_demand_units": 521, "batch_size_units": 10},
    {"product": "P78", "profit_per_unit": 6.48, "r1_per_unit": 1.11, "r2_per_unit": 1.85, "r3_per_unit": 2.05, "upper_demand_units": 501, "batch_size_units": 10},
    {"product": "P79", "profit_per_unit": 11.59, "r1_per_unit": 2.31, "r2_per_unit": 3.78, "r3_per_unit": 2.2, "upper_demand_units": 405, "batch_size_units": 10},
    {"product": "P80", "profit_per_unit": 7.01, "r1_per_unit": 1.29, "r2_per_unit": 0.98, "r3_per_unit": 2.45, "upper_demand_units": 45, "batch_size_units": 10},
    {"product": "P81", "profit_per_unit": 10.08, "r1_per_unit": 4.43, "r2_per_unit": 1.69, "r3_per_unit": 2.7, "upper_demand_units": 431, "batch_size_units": 10},
    {"product": "P82", "profit_per_unit": 9.24, "r1_per_unit": 3.42, "r2_per_unit": 0.9, "r3_per_unit": 1.21, "upper_demand_units": 392, "batch_size_units": 10},
    {"product": "P83", "profit_per_unit": 10.27, "r1_per_unit": 2.19, "r2_per_unit": 3.74, "r3_per_unit": 1.31, "upper_demand_units": 234, "batch_size_units": 10},
    {"product": "P84", "profit_per_unit": 9.34, "r1_per_unit": 1.07, "r2_per_unit": 3.57, "r3_per_unit": 0.55, "upper_demand_units": 549, "batch_size_units": 10},
    {"product": "P85", "profit_per_unit": 9.03, "r1_per_unit": 2.11, "r2_per_unit": 1.4, "r3_per_unit": 1.86, "upper_demand_units": 235, "batch_size_units": 10},
    {"product": "P86", "profit_per_unit": 9.39, "r1_per_unit": 2.17, "r2_per_unit": 2.81, "r3_per_unit": 0.4, "upper_demand_units": 552, "batch_size_units": 10},
    {"product": "P87", "profit_per_unit": 10.6, "r1_per_unit": 3.86, "r2_per_unit": 3.36, "r3_per_unit": 1.56, "upper_demand_units": 413, "batch_size_units": 10},
    {"product": "P88", "profit_per_unit": 9.36, "r1_per_unit": 3.48, "r2_per_unit": 2.44, "r3_per_unit": 1.77, "upper_demand_units": 453, "batch_size_units": 10},
    {"product": "P89", "profit_per_unit": 11.46, "r1_per_unit": 4.53, "r2_per_unit": 2.35, "r3_per_unit": 1.07, "upper_demand_units": 161, "batch_size_units": 10},
    {"product": "P90", "profit_per_unit": 9.83, "r1_per_unit": 2.78, "r2_per_unit": 1.35, "r3_per_unit": 1.9, "upper_demand_units": 178, "batch_size_units": 10},
    {"product": "P91", "profit_per_unit": 7.57, "r1_per_unit": 1.3, "r2_per_unit": 0.83, "r3_per_unit": 0.38, "upper_demand_units": 102, "batch_size_units": 10},
    {"product": "P92", "profit_per_unit": 10.35, "r1_per_unit": 3.8, "r2_per_unit": 3.64, "r3_per_unit": 0.4, "upper_demand_units": 601, "batch_size_units": 10},
    {"product": "P93", "profit_per_unit": 11.9, "r1_per_unit": 4.0, "r2_per_unit": 3.65, "r3_per_unit": 2.52, "upper_demand_units": 217, "batch_size_units": 10},
    {"product": "P94", "profit_per_unit": 10.74, "r1_per_unit": 3.16, "r2_per_unit": 2.72, "r3_per_unit": 1.27, "upper_demand_units": 193, "batch_size_units": 10},
    {"product": "P95", "profit_per_unit": 8.96, "r1_per_unit": 4.04, "r2_per_unit": 1.69, "r3_per_unit": 0.64, "upper_demand_units": 124, "batch_size_units": 10},
    {"product": "P96", "profit_per_unit": 10.45, "r1_per_unit": 2.87, "r2_per_unit": 1.72, "r3_per_unit": 1.71, "upper_demand_units": 257, "batch_size_units": 10},
    {"product": "P97", "profit_per_unit": 11.87, "r1_per_unit": 3.0, "r2_per_unit": 3.04, "r3_per_unit": 2.38, "upper_demand_units": 247, "batch_size_units": 10},
    {"product": "P98", "profit_per_unit": 9.65, "r1_per_unit": 2.6, "r2_per_unit": 3.64, "r3_per_unit": 0.88, "upper_demand_units": 273, "batch_size_units": 10},
    {"product": "P99", "profit_per_unit": 9.84, "r1_per_unit": 0.91, "r2_per_unit": 3.6, "r3_per_unit": 1.98, "upper_demand_units": 271, "batch_size_units": 10},
    {"product": "P100", "profit_per_unit": 8.84, "r1_per_unit": 1.25, "r2_per_unit": 3.23, "r3_per_unit": 0.53, "upper_demand_units": 576, "batch_size_units": 10}
  ],
  "resources": [
    {"resource": "R1", "capacity": 27380.54},
    {"resource": "R2", "capacity": 22245.11},
    {"resource": "R3", "capacity": 15147.73}
  ]
}
```

##### Variable and Parameter Summary

- $x_i$: integer number of batches of product $i$ to produce ($i=1,\ldots,100$)
- $profit\_per\_unit_i$, $r1\_per\_unit_i$, $r2\_per\_unit_i$, $r3\_per\_unit_i$, $upper\_demand\_units_i$, $batch\_size\_units=10$ (all as above)
- $capacity_1=27380.54$, $capacity_2=22245.11$, $capacity_3=15147.73$

##### Full Mathematical Model

$$
\begin{align*}
\max \quad & \sum_{i=1}^{100} \left( 10 \cdot x_i \cdot profit\_per\_unit_i \right) \\
\text{s.t.} \quad
& \sum_{i=1}^{100} 10 \cdot x_i \cdot r1\_per\_unit_i \leq 27380.54 \\
& \sum_{i=1}^{100} 10 \cdot x_i \cdot r2\_per\_unit_i \leq 22245.11 \\
& \sum_{i=1}^{100} 10 \cdot x_i \cdot r3\_per\_unit_i \leq 15147.73 \\
& 10 \cdot x_i \leq upper\_demand\_units_i \quad \forall i=1,\ldots,100 \\
& x_i \in \mathbb{Z}_+, \quad x_i \geq 0 \quad \forall i=1,\ldots,100
\end{align*}
$$

All coefficients and bounds are as retrieved above.