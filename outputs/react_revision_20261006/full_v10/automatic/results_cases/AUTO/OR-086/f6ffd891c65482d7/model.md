Mathematical Optimization Model

Sets:
G = {I, II, III}         // Raw grades (from file_0_view_0, column 'Grade')
B = {Red, Yellow, Blue}  // Wine brands (from file_1_view_0, column 'Brand')

Parameters:
S_g : Daily supply limit of grade g (kg)           // file_0_view_0, 'Daily Supply (kg)'
C_g : Unit cost of grade g (CNY/kg)                // file_0_view_0, 'Cost (CNY/kg)'
P_b : Selling price of brand b (CNY/kg)            // file_1_view_0, 'Selling Price (CNY/kg)'

Blending requirements (from file_1_view_0, 'Blending Requirements'):
For each brand b, let L_{b,g} and U_{b,g} be the lower and upper bounds on the proportion of grade g in brand b.
If no bound is specified, set L_{b,g} = 0, U_{b,g} = 1.

Red:    I < 10% (U_{Red,I}=0.10), II > 50% (L_{Red,II}=0.50)
Yellow: III < 70% (U_{Yellow,III}=0.70), I > 20% (L_{Yellow,I}=0.20)
Blue:   I < 50% (U_{Blue,I}=0.50), II > 10% (L_{Blue,II}=0.10)

Decision Variables:
x_{b,g} ≥ 0 : Amount (kg) of grade g used in brand b

Auxiliary:
y_b = ∑_{g∈G} x_{b,g} : Total production of brand b (kg)

Objective:
Maximize total net profit:
max ∑_{b∈B} P_b y_b - ∑_{g∈G} C_g (∑_{b∈B} x_{b,g})

Constraints:

1. Blending Requirements (for all b∈B, g∈G with specified bounds):
   For all lower bounds L_{b,g}:
      x_{b,g} ≥ L_{b,g} y_b
   For all upper bounds U_{b,g}:
      x_{b,g} ≤ U_{b,g} y_b

   Specifically:
   - Red:
       x_{Red,I} ≤ 0.10 y_{Red}
       x_{Red,II} ≥ 0.50 y_{Red}
   - Yellow:
       x_{Yellow,III} ≤ 0.70 y_{Yellow}
       x_{Yellow,I} ≥ 0.20 y_{Yellow}
   - Blue:
       x_{Blue,I} ≤ 0.50 y_{Blue}
       x_{Blue,II} ≥ 0.10 y_{Blue}

2. Raw Material Supply (for all g∈G):
   ∑_{b∈B} x_{b,g} ≤ S_g

3. Minimum Production for Red:
   y_{Red} ≥ 2000

4. Non-negativity:
   x_{b,g} ≥ 0   for all b∈B, g∈G

Data Mapping:
- file_0_view_0: Grade (G), 'Daily Supply (kg)' (S_g), 'Cost (CNY/kg)' (C_g)
- file_1_view_0: Brand (B), 'Selling Price (CNY/kg)' (P_b), 'Blending Requirements' (L_{b,g}, U_{b,g} as above)