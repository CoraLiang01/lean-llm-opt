##### Objective Function:

$\quad \max \left( 443\,x_{\text{Queens}} + 522\,x_{\text{Brooklyn}} + 300\,x_{\text{Manhattan}} + 767\,x_{\text{Bronx}} + 300\,x_{\text{Staten Island}} + 309\,x_{\text{Harlem}} + 598\,x_{\text{Upper East Side}} + 460\,x_{\text{Lower Manhattan}} + 318\,x_{\text{Midtown}} + 126\,x_{\text{Long Island City}} + 593\,x_{\text{Williamsburg}} + 871\,x_{\text{Bushwick}} + 858\,x_{\text{Flatbush}} + 321\,x_{\text{Greenpoint}} + 275\,x_{\text{Park Slope}} + 700\,x_{\text{Astoria}} + 685\,x_{\text{Jackson Heights}} + 940\,x_{\text{Flushing}} + 522\,x_{\text{Sunnyside}} + 763\,x_{\text{Ditmars}} \right)$

##### Constraints

###### 1. Capacity Constraint:

$104\,x_{\text{Queens}} + 368\,x_{\text{Brooklyn}} + 483\,x_{\text{Manhattan}} + 165\,x_{\text{Bronx}} + 105\,x_{\text{Staten Island}} + 123\,x_{\text{Harlem}} + 131\,x_{\text{Upper East Side}} + 341\,x_{\text{Lower Manhattan}} + 258\,x_{\text{Midtown}} + 469\,x_{\text{Long Island City}} + 387\,x_{\text{Williamsburg}} + 425\,x_{\text{Bushwick}} + 482\,x_{\text{Flatbush}} + 495\,x_{\text{Greenpoint}} + 305\,x_{\text{Park Slope}} + 377\,x_{\text{Astoria}} + 318\,x_{\text{Jackson Heights}} + 56\,x_{\text{Flushing}} + 213\,x_{\text{Sunnyside}} + 472\,x_{\text{Ditmars}} \leq 4466$

###### 2. Variable Constraints:

$x_i \in \mathbb{Z}_{\geq 0}$ for all areas $i$ (i.e., all $x_i$ are non-negative integers)

##### Retrieved Information

{
  "capacity": 4466,
  "areas": [
    {"name": "Queens", "value": 443, "weight": 104},
    {"name": "Brooklyn", "value": 522, "weight": 368},
    {"name": "Manhattan", "value": 300, "weight": 483},
    {"name": "Bronx", "value": 767, "weight": 165},
    {"name": "Staten Island", "value": 300, "weight": 105},
    {"name": "Harlem", "value": 309, "weight": 123},
    {"name": "Upper East Side", "value": 598, "weight": 131},
    {"name": "Lower Manhattan", "value": 460, "weight": 341},
    {"name": "Midtown", "value": 318, "weight": 258},
    {"name": "Long Island City", "value": 126, "weight": 469},
    {"name": "Williamsburg", "value": 593, "weight": 387},
    {"name": "Bushwick", "value": 871, "weight": 425},
    {"name": "Flatbush", "value": 858, "weight": 482},
    {"name": "Greenpoint", "value": 321, "weight": 495},
    {"name": "Park Slope", "value": 275, "weight": 305},
    {"name": "Astoria", "value": 700, "weight": 377},
    {"name": "Jackson Heights", "value": 685, "weight": 318},
    {"name": "Flushing", "value": 940, "weight": 56},
    {"name": "Sunnyside", "value": 522, "weight": 213},
    {"name": "Ditmars", "value": 763, "weight": 472}
  ]
}