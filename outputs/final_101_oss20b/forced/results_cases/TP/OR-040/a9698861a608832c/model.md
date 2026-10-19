##### Decision Variables

$x_i \in \mathbb{Z}_{\ge 0}$: scale of development in area $i$ each day, for $i\in I$.

##### Objective Function

$\max \; 443\,x_{\text{Queens}} + 522\,x_{\text{Brooklyn}} + 300\,x_{\text{Manhattan}} + 767\,x_{\text{Bronx}} + 300\,x_{\text{Staten Island}} + 309\,x_{\text{Harlem}} + 598\,x_{\text{Upper East Side}} + 460\,x_{\text{Lower Manhattan}} + 318\,x_{\text{Midtown}} + 126\,x_{\text{Long Island City}} + 593\,x_{\text{Williamsburg}} + 871\,x_{\text{Bushwick}} + 858\,x_{\text{Flatbush}} + 321\,x_{\text{Greenpoint}} + 275\,x_{\text{Park Slope}} + 700\,x_{\text{Astoria}} + 685\,x_{\text{Jackson Heights}} + 940\,x_{\text{Flushing}} + 522\,x_{\text{Sunnyside}} + 763\,x_{\text{Ditmars}}$

##### Constraints

1. Capacity: $104\,x_{\text{Queens}} + 368\,x_{\text{Brooklyn}} + 483\,x_{\text{Manhattan}} + 165\,x_{\text{Bronx}} + 105\,x_{\text{Staten Island}} + 123\,x_{\text{Harlem}} + 131\,x_{\text{Upper East Side}} + 341\,x_{\text{Lower Manhattan}} + 258\,x_{\text{Midtown}} + 469\,x_{\text{Long Island City}} + 387\,x_{\text{Williamsburg}} + 425\,x_{\text{Bushwick}} + 482\,x_{\text{Flatbush}} + 495\,x_{\text{Greenpoint}} + 305\,x_{\text{Park Slope}} + 377\,x_{\text{Astoria}} + 318\,x_{\text{Jackson Heights}} + 56\,x_{\text{Flushing}} + 213\,x_{\text{Sunnyside}} + 472\,x_{\text{Ditmars}} \;\le\; 4466$

2. Non‑negativity: $x_i \ge 0$, integer, for all $i\in I$.

$I=\{\text{Queens},\text{Brooklyn},\text{Manhattan},\text{Bronx},\text{Staten Island},\text{Harlem},\text{Upper East Side},\text{Lower Manhattan},\text{Midtown},\text{Long Island City},\text{Williamsburg},\text{Bushwick},\text{Flatbush},\text{Greenpoint},\text{Park Slope},\text{Astoria},\text{Jackson Heights},\text{Flushing},\text{Sunnyside},\text{Ditmars}\}$

###### Retrieved Information

{
  "capacity": 4466,
  "products": [
    {"ProductName":"Queens","Value":443,"Weight":104},
    {"ProductName":"Brooklyn","Value":522,"Weight":368},
    {"ProductName":"Manhattan","Value":300,"Weight":483},
    {"ProductName":"Bronx","Value":767,"Weight":165},
    {"ProductName":"Staten Island","Value":300,"Weight":105},
    {"ProductName":"Harlem","Value":309,"Weight":123},
    {"ProductName":"Upper East Side","Value":598,"Weight":131},
    {"ProductName":"Lower Manhattan","Value":460,"Weight":341},
    {"ProductName":"Midtown","Value":318,"Weight":258},
    {"ProductName":"Long Island City","Value":126,"Weight":469},
    {"ProductName":"Williamsburg","Value":593,"Weight":387},
    {"ProductName":"Bushwick","Value":871,"Weight":425},
    {"ProductName":"Flatbush","Value":858,"Weight":482},
    {"ProductName":"Greenpoint","Value":321,"Weight":495},
    {"ProductName":"Park Slope","Value":275,"Weight":305},
    {"ProductName":"Astoria","Value":700,"Weight":377},
    {"ProductName":"Jackson Heights","Value":685,"Weight":318},
    {"ProductName":"Flushing","Value":940,"Weight":56},
    {"ProductName":"Sunnyside","Value":522,"Weight":213},
    {"ProductName":"Ditmars","Value":763,"Weight":472}
  ]
}