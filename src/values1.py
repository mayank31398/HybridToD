ALLOWED_DOMAINS = {
    "hotel",
    "restaurant",
    "attraction"
}
ALLOWED_SLOTS = {
    "address",
    "area",
    "internet",
    "parking",
    # "id",
    # "location",
    "name",
    "phone",
    "postcode",
    # "price",
    "pricerange",
    "stars",
    "takesbookings",
    "type",
    # "n",
    "entrance fee",
    "openhours",
    "food",
    # "introduction",
    "signature"
}
NAME_MAP = {
    "the university arms": "university arms hotel",
    "the gonville": "gonville hotel",
    "the lovell": "lovell lodge",
    "lovell hotel": "lovell lodge",
    "the lovell hotel": "lovell lodge",
    "the huntingdon marriott": "huntingdon marriott hotel",
    "junction theatre": "the junction",
    "the acorn house": "acorn guest house",
    "the acorn": "acorn guest house",
    "acorn house": "acorn guest house",
    "cafe gallery": "cafe jello gallery",
    "the peking": "peking restaurant",
    "cb2311": "cb23ll",
    "nashua": "nusha",
    "a&b guesthouse": "a and b guest house",
    # "a&b guesthouse": "a and b guest house",
    "alexander b&b": "alexander bed and breakfast",
    "alexander b and b": "alexander bed and breakfast",
    "alexander b & b.": "alexander bed and breakfast",
    "the alexander b&b": "alexander bed and breakfast",
    "the botanic gardens": "cambridge university botanic gardens",
    "ian hong house restaurant": "lan hong house",
    "the alpha - milton": "alpha-milton guest house",
    "the carolina": "carolina bed and breakfast"
}
SORTED_KEYS = [
    "hotel-address",
    "restaurant-address",
    "attraction-address",

    "hotel-name",
    "restaurant-name",
    "attraction-name",

    "hotel-phone",
    "restaurant-phone",
    "attraction-phone",

    "hotel-postcode",
    "restaurant-postcode",
    "attraction-postcode",

    "hotel-area",
    "restaurant-area",
    "attraction-area",

    "restaurant-food",

    "hotel-pricerange",
    "restaurant-pricerange",
    "attraction-pricerange",

    "attraction-entrance fee",

    "hotel-type",
    "restaurant-type",
    "attraction-type",

    "hotel-internet",
    "hotel-parking",
    "hotel-stars",
    "hotel-takesbookings",
    "restaurant-signature",
    "attraction-openhours"
]

ALL_DOMAINS = ["restaurant", "hotel", "attraction",
               "train", "taxi", "police", "hospital"]
SLOT_NORMALIZE = {
    "car type": "car",
    "entrance fee": "price",
    "leaveat": "leave",
    "arriveby": "arrive",
    "trainid": "id",
    "addr": "address",
    "post": "postcode",
    "ref": "reference",
    "fee": "price",
    "ticket": "price",
    "price range": "pricerange",
    "price": "pricerange",
    "depart": "departure",
    "dest": "destination",
}
SKIP_CASE = {
    "don't care": 1,
    "do n't care": 1,
    "dont care": 1,
    "not mentioned": 1,
    "dontcare": 1,
    "": 1
}
INFORMABLE_SLOTS = {
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people", "sfek"],
    "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name", "sfek"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave", "sfek"],
    "taxi": ["leave", "destination", "departure", "arrive", "sfek"],
    "police": [],
    "hospital": ["department"],
}
REQUESTABLE_SLOTS = {
    "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area", "reference", "name"],
    "hotel": ["address", "postcode", "internet", "phone", "parking", "pricerange", "stars", "area", "reference", "name"],
    "attraction": ["price", "type", "address", "postcode", "phone", "area", "reference", "name", "pricerange"],
    "train": ["duration", "leave", "price", "arrive", "id", "reference"],
    "taxi": ["car", "phone"],
    "police": ["postcode", "address", "phone"],
    "hospital": ["address", "phone", "postcode"]
}
BOOLEAN_STRINGS = ["yes", "no"]
SPECIAL_TOKENS = {
    "domain": "<domain>",
    "entity_name": "<entity_name>",
    "bos": "<s>",
    "eos": "</s>",
    "pad": "<pad>",
    "user": "<user>",
    "system": "<system>",
    "kb": "<kb>",
    "doc": "<doc>",
    "doc_topic": "<doc_topic>",
    "doc_title": "<doc_title>",
    "doc_body": "<doc_body>",
    "db": "<db>",
    "slot": "<slot>",
    "value": "<value>",
    "dict_slot": "<dict_slot>",
    "dict_value": "<dict_value>",
    "list_value": "<list_value>",
    "entity_label": "<entity_label>",
    "entity_explanation": "<entity_explanation>",
    "task_entity_selection": ":entity_selection:",
    "task_response_generation": ":response_generation:"
}
UNIQUES = {
    "restaurant": ["phone", "postcode", "address"],
    "hotel": ["address", "postcode", "phone"],
    "attraction": ["address", "postcode", "phone"]
}
UNIQUE_SLOTS = ["address", "phone", "postcode"]
