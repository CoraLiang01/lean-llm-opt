Let \( x_p \geq 0 \) be the continuous production quantity of product \( p \) for each \( p \in \{\text{P1}, \text{P2}, \ldots, \text{P111}\} \).

Parameters:
- \( c_p \): Unit profit of product \( p \), from unit_product_profits.csv.
- \( a_{d,p} \): Processing time required on device \( d \) per unit of product \( p \), from device_time.csv.
- \( b_d \): Monthly capacity of device \( d \), from monthly_device_capacity.csv.

Data (in supplied order):

unit_product_profits.csv:
(Product, Unit_Profit)
P1, 28.55
P2, 12.78
P3, 45.21
P4, 18.92
P5, 33.47
P6, 8.64
P7, 25.88
P8, 40.15
P9, 14.39
P10, 37.62
P11, 10.25
P12, 29.81
P13, 48.77
P14, 16.53
P15, 42.09
P16, 9.11
P17, 22.46
P18, 36.73
P19, 11.99
P20, 31.28
P21, 7.45
P22, 27.6
P23, 43.91
P24, 15.82
P25, 39.36
P26, 6.23
P27, 20.71
P28, 34.98
P29, 13.24
P30, 32.53
P31, 5.1
P32, 24.18
P33, 47.85
P34, 19.3
P35, 41.44
P36, 11.02
P37, 26.57
P38, 38.2
P39, 17.76
P40, 46.12
P41, 9.89
P42, 21.34
P43, 33.86
P44, 14.88
P45, 30.41
P46, 6.5
P47, 28.93
P48, 49.2
P49, 18.15
P50, 44.78
P51, 10.57
P52, 23.69
P53, 35.51
P54, 16.03
P55, 38.83
P56, 5.88
P57, 29.45
P58, 42.33
P59, 12.41
P60, 37.06
P61, 7.99
P62, 21.9
P63, 46.99
P64, 19.87
P65, 40.7
P66, 8.34
P67, 26.11
P68, 39.54
P69, 14.07
P70, 35.79
P71, 6.92
P72, 23.03
P73, 45.56
P74, 17.2
P75, 43.27
P76, 9.52
P77, 28.08
P78, 41.1
P79, 15.46
P80, 34.3
P81, 5.43
P82, 20.25
P83, 48.38
P84, 18.68
P85, 40.01
P86, 11.45
P87, 25.32
P88, 37.58
P89, 13.62
P90, 31.84
P91, 7.27
P92, 24.75
P93, 49.88
P94, 16.85
P95, 42.82
P96, 10.13
P97, 27.36
P98, 36.19
P99, 12.8
P100, 30.09
P101, 6.07
P102, 22.67
P103, 44.42
P104, 19.54
P105, 47.41
P106, 8.78
P107, 29.97
P108, 38.96
P109, 15.11
P110, 33.05
P111, 9.99

device_time.csv:
For each device \( d \in \{\text{A},\ldots,\text{J}\} \), and each product \( p \in \{\text{P1},\ldots,\text{P111}\} \), the processing time \( a_{d,p} \) is as given in the corresponding row and column.

monthly_device_capacity.csv:
(Device, Monthly_Capacity)
A, 3500
B, 4200
C, 4500
D, 2800
E, 3300
F, 3800
G, 4100
H, 3900
I, 4800
J, 3100

Model:

Variables:
For each product \( p \in \{\text{P1},\ldots,\text{P111}\} \):
\[
x_p \geq 0 \quad \text{(continuous)}
\]

Objective:
\[
\max \sum_{p} c_p x_p
\]
where \( c_p \) is the Unit_Profit for product \( p \) from unit_product_profits.csv.

Constraints:
For each device \( d \in \{\text{A},\ldots,\text{J}\} \):
\[
\sum_{p} a_{d,p} x_p \leq b_d
\]
where \( a_{d,p} \) is the processing time of product \( p \) on device \( d \) from device_time.csv, and \( b_d \) is the Monthly_Capacity for device \( d \) from monthly_device_capacity.csv.

Explicitly, for each device (in the order A, B, C, D, E, F, G, H, I, J):

For device \( d \) (e.g., A):
\[
\sum_{p=\text{P1}}^{\text{P111}} a_{\text{A},p} x_p \leq 3500
\]
and similarly for the other devices, using their respective \( a_{d,p} \) and \( b_d \).

Domain:
\[
x_p \geq 0 \quad \forall p \in \{\text{P1},\ldots,\text{P111}\}
\]

This is a continuous linear program with 111 variables and 10 constraints, using all coefficients and identifiers as supplied in the CSVs.