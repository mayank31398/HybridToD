from typing import List
import numpy as np
from dataset1 import DB, KB
from values1 import ALLOWED_SLOTS
import matplotlib.pyplot as plt

# y = np.array([33, 110, 79])
# mylabels = ["Hotel", "Restaurant", "Attraction"]

# plt.pie(y, labels = mylabels, startangle = 90, colors=["c", "m", "y"],
#         textprops={"fontsize": 14},
#         autopct=lambda p: '{:.1f}%'.format(p),
#         wedgeprops={"edgecolor":"k",'linewidth': 1, 'linestyle': 'solid', 'antialiased': True})
# plt.tight_layout()
# plt.savefig("pie.pdf", dpi=1200)
# plt.show()

def GetTicks(d: dict, datasets: List[str], domain: str) -> List[List[int]]:
    xticks = []
    for ds in d[datasets[0]]:
        d_, s = ds.split("-")
        if (d_ == domain):
            xticks.append(s)
    xticks.sort()

    yticks = []
    for dataset in datasets:
        yticks.append([])
        for slot in xticks:
            if (domain + "-" + slot in d[dataset]):
                yticks[-1].append(d[dataset][domain + "-" + slot])
            else:
                yticks[-1].append(0)

    return xticks, yticks


def Draw(d: dict, datasets: List[str], domain: str):
    xticks, yticks = GetTicks(d, datasets, domain)

    barWidth = 0.25
    _, ax = plt.subplots(figsize =(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(xticks))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    ax.bar(br1, yticks[0], color ='c', width = barWidth,
        edgecolor ='grey', label="SeKnow-MultiWoZ")
    ax.bar(br2, yticks[1], color ='m', width = barWidth,
        edgecolor ='grey', label="HybridToD")
    # ax.bar(br3, yticks[2], color ='y', width = barWidth,
    #     edgecolor ='grey', label="UnstructuredToD")
    # for bars in ax.containers:
    #     ax.bar_label(bars)

    # Adding Xticks
    plt.xlabel('Slot type', fontsize=20)
    plt.ylabel('Total number of entities', fontsize=20)
    plt.xticks([r + barWidth for r in range(len(xticks))], xticks, fontsize=20, rotation="vertical")
    plt.yticks(fontsize=20)
    plt.title(domain[0].upper() + domain[1:].lower(), fontsize=14)

    plt.tight_layout()
    plt.legend()

    plt.savefig(domain + ".pdf", dpi=1200)
    plt.show()


datasets = [
    "datasets_allowed_domains/preprocessed_data",
    "datasets_protocol1/preprocessed_data",
    "datasets_fully_unstructured/preprocessed_data"
]

d = {}
for dataset in datasets:
    db = DB(dataset)
    kb = KB(dataset)
    d[dataset] = {}

    for domain, entity_name, db_entity in db:
        kb_entity = kb.GetEntity(domain, entity_name)

        for slot_value in db_entity:
            if (slot_value.slot in ALLOWED_SLOTS):
                if (domain + "-" + slot_value.slot not in d[dataset]):
                    d[dataset][domain + "-" + slot_value.slot] = 0
                d[dataset][domain + "-" + slot_value.slot] += 1

Draw(d, datasets, "restaurant")



# d = {}
# db = DB(datasets[0])
# kb = KB(datasets[0])
# for domain, entity_name, db_entity in db:
#     kb_entity = kb.GetEntity(domain, entity_name)

#     if (domain not in d):
#         d[domain] = {
#             "db": 0,
#             "kb": 0,
#             "count": 0
#         }

#     d[domain]["count"] += 1

#     n = 0
#     for slot_value in db_entity:
#         if (slot_value.slot in ALLOWED_SLOTS):
#             n += 1
#     d[domain]["db"] += n

#     d[domain]["kb"] += len(kb_entity.entity)

# for domain in d:
#     d[domain]["db"] /= d[domain]["count"]
#     d[domain]["kb"] /= d[domain]["count"]

# print(d)
