{
  "query": "A utility company must place emergency repair depots so that every service zone is covered by at least one selected depot. Candidate depot opening costs and covered service zones are listed in facility_sites.csv, and the full set of service zones is listed in service_zones.csv.\n\nFormulate a minimum-cost set covering model. For each candidate depot i, define y_i as a binary variable equal to 1 if depot i is opened. The objective is to minimize total opening cost. The model should include one coverage constraint for each service zone requiring at least one opened depot that covers that zone, and binary restrictions for all depot-opening variables.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Center",
        "OpeningCost",
        "CoveredDistricts"
      ],
      "file_name": "facility_sites.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Center": "B1",
            "CoveredDistricts": "Z1;Z2;Z5",
            "OpeningCost": "11"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Center": "B2",
            "CoveredDistricts": "Z2;Z3;Z6",
            "OpeningCost": "14"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Center": "B3",
            "CoveredDistricts": "Z4;Z5;Z8",
            "OpeningCost": "10"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Center": "B4",
            "CoveredDistricts": "Z1;Z6;Z7",
            "OpeningCost": "13"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Center": "B5",
            "CoveredDistricts": "Z3;Z7;Z9",
            "OpeningCost": "16"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Center": "B6",
            "CoveredDistricts": "Z8;Z9;Z10",
            "OpeningCost": "9"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Center": "B7",
            "CoveredDistricts": "Z4;Z10",
            "OpeningCost": "12"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Center": "B8",
            "CoveredDistricts": "Z5;Z6;Z9",
            "OpeningCost": "15"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "Zone"
      ],
      "file_name": "service_zones.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Zone": "Z1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Zone": "Z2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Zone": "Z3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Zone": "Z4"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Zone": "Z5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Zone": "Z6"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Zone": "Z7"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Zone": "Z8"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Zone": "Z9"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "Zone": "Z10"
          }
        }
      ],
      "table_id": "file_1_view_0"
    }
  ]
}
