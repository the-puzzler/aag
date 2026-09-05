#!/usr/bin/env python
"""Coarse groupings of the 1000 ImageNet classes from the WordNet hierarchy.

Why this exists: the assignment must make z independent of the CONDITION, and
the project's standing lesson (VPT action marginals, 24-frame context) is that
independence from the joint condition does not imply independence from any
low-dimensional function of it. For ImageNet the joint condition is the 1000-way
class; the low-dimensional functions a generator can key on are the coarse
categories -- "is a dog", "is an artifact", "is a bird". Transporting per exact
class (~1281 members) leaves those coarse partitions untouched, exactly as the
81-way action transport left "turning vs not" at 1.27x. So the assignment
interleaves group transport over several hierarchy levels, and this script
produces those levels.

Groups at depth k are the k-th ancestor along each class synset's PRIMARY
hypernym chain (hypernyms()[0] repeated), which is a tree and therefore a
proper partition. WordNet is a DAG in general; the primary chain is the same
choice torchvision/BREEDS-style tooling makes for a single lineage.

Output: a .pt with {"wnids", "names", "levels": {name: LongTensor(1000)},
"level_sizes": {name: n_groups}} plus a human-readable summary. Class index i
is the standard ILSVRC-2012 index (sorted wnid order), which is also the HF
mirror's label index.
"""
from __future__ import annotations

import argparse, json, os
from collections import Counter
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--class-index", default=os.environ.get("IMAGENET_CLASS_INDEX",
                "/home/ubuntu/.claude/jobs/1c9fe4d1/tmp/imagenet_class_index.json"))
ap.add_argument("--nltk-data", default="/data/tmp/nltk_data")
ap.add_argument("--depths", default="4,5,6,7,8,9,10",
                help="ancestor depths from the root to emit as levels")
ap.add_argument("--out", type=Path, default=Path("/data/aag_data/imagenet256/class_groups.pt"))
a = ap.parse_args()

os.environ["NLTK_DATA"] = a.nltk_data
from nltk.corpus import wordnet as wn  # noqa: E402

idx = json.load(open(a.class_index))
assert len(idx) == 1000
wnids = [idx[str(i)][0] for i in range(1000)]
names = [idx[str(i)][1] for i in range(1000)]
# sanity: the standard index is sorted-wnid order
assert wnids == sorted(wnids), "class index is not in sorted-wnid order"

def chain(wnid):
    s = wn.synset_from_pos_and_offset("n", int(wnid[1:]))
    out = [s]
    while s.hypernyms() or s.instance_hypernyms():
        hs = s.hypernyms() or s.instance_hypernyms()
        s = hs[0]
        out.append(s)
    return out[::-1]          # root first: entity.n.01, ..., class

chains = [chain(w) for w in wnids]
maxd = max(len(c) for c in chains)
print(f"chain depth: min {min(len(c) for c in chains)}  max {maxd}")

levels, sizes, summary = {}, {}, {}
for d in [int(x) for x in a.depths.split(",")]:
    # a class shallower than d is its own group (its leaf), so every class has one
    anc = [c[min(d, len(c) - 1)].name() for c in chains]
    uniq = sorted(set(anc))
    ids = torch.tensor([uniq.index(x) for x in anc], dtype=torch.long)
    cnt = Counter(anc)
    levels[f"depth{d}"] = ids
    sizes[f"depth{d}"] = len(uniq)
    top = cnt.most_common(6)
    summary[f"depth{d}"] = {"n_groups": len(uniq), "largest": top,
                            "singletons": sum(1 for v in cnt.values() if v == 1)}
    print(f"depth{d:2d}: {len(uniq):4d} groups  largest {top[:4]}  "
          f"singletons {summary[f'depth{d}']['singletons']}")

# two named coarse splits that are obvious generator handles
def has(c, *names_):
    return any(s.name() in names_ for s in c)
living = torch.tensor([int(has(c, "living_thing.n.01")) for c in chains])
animal = torch.tensor([int(has(c, "animal.n.01")) for c in chains])
dog = torch.tensor([int(has(c, "dog.n.01")) for c in chains])
levels["living"] = living; sizes["living"] = 2
levels["animal"] = animal; sizes["animal"] = 2
levels["dog"] = dog; sizes["dog"] = 2
print(f"living things {int(living.sum())}/1000, animals {int(animal.sum())}, dogs {int(dog.sum())}")
levels["joint"] = torch.arange(1000); sizes["joint"] = 1000

a.out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"wnids": wnids, "names": names, "levels": levels, "level_sizes": sizes,
            "chains": [[s.name() for s in c] for c in chains]}, a.out)
(a.out.with_suffix(".json")).write_text(json.dumps(summary, indent=1))
print("saved", a.out)
