##### Objective Function:

$\quad \max \sum_{i \in \mathcal{A}} v_i x_i$

where $\mathcal{A}$ is the set of areas, $v_i$ is the benefit coefficient for area $i$, and $x_i$ is the integer scale of development in area $i$ per day.

##### Constraints:

$\sum_{i \in \mathcal{A}} w_i x_i \leq 4466$

$x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{A}$

##### Retrieved Information

{
  "areas": [
    "Queens",
    "Brooklyn",
    "Manhattan",
    "Bronx",
    "Staten Island",
    "Harlem",
    "Upper East Side",
    "Lower Manhattan",
    "Midtown",
    "Long Island City",
    "Williamsburg",
    "Bushwick",
    "Flatbush",
    "Greenpoint",
    "Park Slope",
    "Astoria",
    "Jackson Heights",
    "Flushing",
    "Sunnyside",
    "Ditmars"
  ],
  "benefit_coefficients": {
    "Queens": 443,
    "Brooklyn": 522,
    "Manhattan": 300,
    "Bronx": 767,
    "Staten Island": 300,
    "Harlem": 309,
    "Upper East Side": 598,
    "Lower Manhattan": 460,
    "Midtown": 318,
    "Long Island City": 126,
    "Williamsburg": 593,
    "Bushwick": 871,
    "Flatbush": 858,
    "Greenpoint": 321,
    "Park Slope": 275,
    "Astoria": 700,
    "Jackson Heights": 685,
    "Flushing": 940,
    "Sunnyside": 522,
    "Ditmars": 763
  },
  "weights": {
    "Queens": 104,
    "Brooklyn": 368,
    "Manhattan": 483,
    "Bronx": 165,
    "Staten Island": 105,
    "Harlem": 123,
    "Upper East Side": 131,
    "Lower Manhattan": 341,
    "Midtown": 258,
    "Long Island City": 469,
    "Williamsburg": 387,
    "Bushwick": 425,
    "Flatbush": 482,
    "Greenpoint": 495,
    "Park Slope": 305,
    "Astoria": 377,
    "Jackson Heights": 318,
    "Flushing": 56,
    "Sunnyside": 213,
    "Ditmars": 472
  },
  "capacity": 4466
}

##### Full Model (with explicit variables):

Let $x_{\text{Queens}}, x_{\text{Brooklyn}}, x_{\text{Manhattan}}, x_{\text{Bronx}}, x_{\text{Staten Island}}, x_{\text{Harlem}}, x_{\text{Upper East Side}}, x_{\text{Lower Manhattan}}, x_{\text{Midtown}}, x_{\text{Long Island City}}, x_{\text{Williamsburg}}, x_{\text{Bushwick}}, x_{\text{Flatbush}}, x_{\text{Greenpoint}}, x_{\text{Park Slope}}, x_{\text{Astoria}}, x_{\text{Jackson Heights}}, x_{\text{Flushing}}, x_{\text{Sunnyside}}, x_{\text{Ditmars}} \in \mathbb{Z}_{\geq 0}$

Maximize:

$443x_{\text{Queens}} + 522x_{\text{Brooklyn}} + 300x_{\text{Manhattan}} + 767x_{\text{Bronx}} + 300x_{\text{Staten Island}} + 309x_{\text{Harlem}} + 598x_{\text{Upper East Side}} + 460x_{\text{Lower Manhattan}} + 318x_{\text{Midtown}} + 126x_{\text{Long Island City}} + 593x_{\text{Williamsburg}} + 871x_{\text{Bushwick}} + 858x_{\text{Flatbush}} + 321x_{\text{Greenpoint}} + 275x_{\text{Park Slope}} + 700x_{\text{Astoria}} + 685x_{\text{Jackson Heights}} + 940x_{\text{Flushing}} + 522x_{\text{Sunnyside}} + 763x_{\text{Ditmars}}$

Subject to:

$104x_{\text{Queens}} + 368x_{\text{Brooklyn}} + 483x_{\text{Manhattan}} + 165x_{\text{Bronx}} + 105x_{\text{Staten Island}} + 123x_{\text{Harlem}} + 131x_{\text{Upper East Side}} + 341x_{\text{Lower Manhattan}} + 258x_{\text{Midtown}} + 469x_{\text{Long Island City}} + 387x_{\text{Williamsburg}} + 425x_{\text{Bushwick}} + 482x_{\text{Flatbush}} + 495x_{\text{Greenpoint}} + 305x_{\text{Park Slope}} + 377x_{\text{Astoria}} + 318x_{\text{Jackson Heights}} + 56x_{\text{Flushing}} + 213x_{\text{Sunnyside}} + 472x_{\text{Ditmars}} \leq 4466$

and

$x_i \in \mathbb{Z}_{\geq 0}$ for all areas $i$.