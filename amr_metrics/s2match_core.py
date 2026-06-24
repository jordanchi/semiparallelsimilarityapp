"""S²MATCH (soft semantic SMATCH) for AMR graph pairs.

Adapted from Opitz et al., 2020 and flipz357/amr-metric-suite.
"""

from __future__ import annotations

import math
import random
import re
from typing import Dict, Tuple

import amr
import numpy as np
import smatch
from scipy.spatial.distance import cityblock, cosine, euclidean

ITERATION_NUM = 5


def load_vecs(fp: str) -> Dict[str, np.ndarray]:
    vectors: Dict[str, np.ndarray] = {}
    with open(fp, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            vectors[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return vectors


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dist = cosine(a, b)
    return 1 - min(1.0, dist)


def euclidean_sim(a: np.ndarray, b: np.ndarray) -> float:
    return 1 / (math.e ** euclidean(a, b))


def cityblock_sim(a: np.ndarray, b: np.ndarray) -> float:
    return 1 / (math.e ** cityblock(a, b))


def maybe_get_vec(word: str, vecs: Dict[str, np.ndarray], mwp: str = "split") -> np.ndarray | None:
    if word in vecs:
        return np.copy(vecs[word])
    if "-" not in word or mwp != "split":
        return None
    partial = [vecs[w] for w in word.split("-") if w in vecs]
    if not partial:
        return None
    return np.sum(np.array(partial), axis=0)


def maybe_sim(
    a: str,
    b: str,
    vecs: Dict[str, np.ndarray],
    cutoff: float = 0.5,
    diffsense: float = 0.5,
    simfun=cosine_sim,
    mwp: str = "split",
) -> float:
    if a == b:
        return 1.0

    a_wo_sense = "-".join(a.split("-")[:-1]) if "-" in a and re.match(r".*[0-9]+", a) else None
    b_wo_sense = "-".join(b.split("-")[:-1]) if "-" in b and re.match(r".*[0-9]+", b) else None

    if a_wo_sense and b_wo_sense and a_wo_sense == b_wo_sense:
        return diffsense
    if a_wo_sense and a_wo_sense == b:
        return diffsense
    if b_wo_sense and b_wo_sense == a:
        return diffsense

    a_vec = maybe_get_vec(a_wo_sense or a, vecs, "None" if a_wo_sense else mwp)
    b_vec = maybe_get_vec(b_wo_sense or b, vecs, "None" if b_wo_sense else mwp)
    if a_vec is None or b_vec is None:
        return 0.0

    if a_wo_sense:
        extra = maybe_get_vec(a_wo_sense + "s", vecs, "None")
        if extra is not None:
            a_vec = a_vec + extra
    if b_wo_sense:
        extra = maybe_get_vec(b_wo_sense + "s", vecs, "None")
        if extra is not None:
            b_vec = b_vec + extra

    sim = simfun(a_vec, b_vec)
    if not sim or sim <= cutoff:
        return 0.0
    if a_wo_sense or b_wo_sense:
        return sim * diffsense
    return sim


def maybe_has_sim(
    a: str,
    b: str,
    sim_dict: Dict[str, float],
    vecs: Dict[str, np.ndarray],
    cutoff: float = 0.5,
    diffsense: float = 0.5,
    simfun=cosine_sim,
    mwp: str = "split",
) -> float:
    key = f"{a}_{b}"
    rev = f"{b}_{a}"
    if key in sim_dict:
        return sim_dict[key]
    if rev in sim_dict:
        return sim_dict[rev]
    score = maybe_sim(a, b, vecs, cutoff=cutoff, diffsense=diffsense, simfun=simfun, mwp=mwp)
    sim_dict[key] = score
    sim_dict[rev] = score
    return score


def compute_pool(
    instance1,
    attribute1,
    relation1,
    instance2,
    attribute2,
    relation2,
    prefix1,
    prefix2,
    vectors,
    cutoff: float = 0.5,
    diffsense: float = 0.5,
    simfun=cosine_sim,
    mwp: str = "split",
):
    candidate_mapping = []
    weight_dict = {}
    sim_dict: Dict[str, float] = {}

    for _ in instance1:
        candidate_mapping.append(set())

    for i, inst1 in enumerate(instance1):
        for j, inst2 in enumerate(instance2):
            if inst1[0].lower() != inst2[0].lower():
                continue
            similarity = maybe_has_sim(
                inst1[2].lower(),
                inst2[2].lower(),
                sim_dict,
                vecs=vectors,
                cutoff=cutoff,
                diffsense=diffsense,
                simfun=simfun,
                mwp=mwp,
            )
            node1_index = int(inst1[1][len(prefix1) :])
            node2_index = int(inst2[1][len(prefix2) :])
            candidate_mapping[node1_index].add(node2_index)
            node_pair = (node1_index, node2_index)
            weight_dict.setdefault(node_pair, {})
            weight_dict[node_pair][-1] = weight_dict[node_pair].get(-1, 0.0) + similarity

    for attr1 in attribute1:
        for attr2 in attribute2:
            if attr1[0].lower() == attr2[0].lower() and attr1[2].lower() == attr2[2].lower():
                node1_index = int(attr1[1][len(prefix1) :])
                node2_index = int(attr2[1][len(prefix2) :])
                candidate_mapping[node1_index].add(node2_index)
                node_pair = (node1_index, node2_index)
                weight_dict.setdefault(node_pair, {})
                weight_dict[node_pair][-1] = weight_dict[node_pair].get(-1, 0.0) + 1.0
            elif attr1[0].lower() == attr2[0].lower() == "top":
                similarity = maybe_has_sim(
                    attr1[2].lower(),
                    attr2[2].lower(),
                    sim_dict,
                    vecs=vectors,
                    cutoff=cutoff,
                    diffsense=diffsense,
                    simfun=simfun,
                    mwp=mwp,
                )
                node1_index = int(attr1[1][len(prefix1) :])
                node2_index = int(attr2[1][len(prefix2) :])
                candidate_mapping[node1_index].add(node2_index)
                node_pair = (node1_index, node2_index)
                weight_dict.setdefault(node_pair, {})
                weight_dict[node_pair][-1] = weight_dict[node_pair].get(-1, 0.0) + similarity

    for rel1 in relation1:
        for rel2 in relation2:
            if rel1[0].lower() != rel2[0].lower():
                continue
            node1_index_amr1 = int(rel1[1][len(prefix1) :])
            node1_index_amr2 = int(rel2[1][len(prefix2) :])
            node2_index_amr1 = int(rel1[2][len(prefix1) :])
            node2_index_amr2 = int(rel2[2][len(prefix2) :])
            candidate_mapping[node1_index_amr1].add(node1_index_amr2)
            candidate_mapping[node2_index_amr1].add(node2_index_amr2)
            node_pair1 = (node1_index_amr1, node1_index_amr2)
            node_pair2 = (node2_index_amr1, node2_index_amr2)
            if node_pair2 == node_pair1:
                weight_dict.setdefault(node_pair1, {})
                weight_dict[node_pair1][-1] = weight_dict[node_pair1].get(-1, 0.0) + 1.0
                continue
            if node1_index_amr1 > node2_index_amr1:
                node_pair1, node_pair2 = node_pair2, node_pair1
            weight_dict.setdefault(node_pair1, {})
            weight_dict[node_pair1].setdefault(-1, 0.0)
            weight_dict[node_pair1][node_pair2] = weight_dict[node_pair1].get(node_pair2, 0.0) + 1.0
            weight_dict.setdefault(node_pair2, {})
            weight_dict[node_pair2].setdefault(-1, 0.0)
            weight_dict[node_pair2][node_pair1] = weight_dict[node_pair2].get(node_pair1, 0.0) + 1.0

    return candidate_mapping, weight_dict


def get_best_match(
    instance1,
    attribute1,
    relation1,
    instance2,
    attribute2,
    relation2,
    prefix1,
    prefix2,
    vectors,
    cutoff: float = 0.5,
    diffsense: float = 0.5,
    simfun=cosine_sim,
    mwp: str = "split",
):
    candidate_mappings, weight_dict = compute_pool(
        instance1,
        attribute1,
        relation1,
        instance2,
        attribute2,
        relation2,
        prefix1,
        prefix2,
        vectors,
        cutoff=cutoff,
        diffsense=diffsense,
        simfun=simfun,
        mwp=mwp,
    )

    best_match_num = 0.0
    best_mapping = [-1] * len(instance1)
    for i in range(ITERATION_NUM):
        if i == 0:
            cur_mapping = smatch.smart_init_mapping(candidate_mappings, instance1, instance2)
        else:
            cur_mapping = smatch.random_init_mapping(candidate_mappings)
        match_num = smatch.compute_match(cur_mapping, weight_dict)
        while True:
            gain, new_mapping = smatch.get_best_gain(
                cur_mapping,
                candidate_mappings,
                weight_dict,
                len(instance2),
                match_num,
            )
            if gain <= 1e-10:
                break
            match_num += gain
            cur_mapping = new_mapping[:]
        if match_num > best_match_num:
            best_mapping = cur_mapping[:]
            best_match_num = match_num
    return best_mapping, best_match_num


def score_pair(
    amr_line1: str,
    amr_line2: str,
    vectors: Dict[str, np.ndarray],
    cutoff: float = 0.5,
    diffsense: float = 0.5,
) -> Tuple[float, float, float]:
    smatch.match_triple_dict.clear()
    amr1 = amr.AMR.parse_AMR_line(amr_line1)
    amr2 = amr.AMR.parse_AMR_line(amr_line2)
    if amr1 is None or amr2 is None:
        return 0.0, 0.0, 0.0

    prefix1, prefix2 = "a", "b"
    amr1.rename_node(prefix1)
    amr2.rename_node(prefix2)
    instance1, attributes1, relation1 = amr1.get_triples()
    instance2, attributes2, relation2 = amr2.get_triples()
    _, best_match_num = get_best_match(
        instance1,
        attributes1,
        relation1,
        instance2,
        attributes2,
        relation2,
        prefix1,
        prefix2,
        vectors,
        cutoff=cutoff,
        diffsense=diffsense,
    )
    test_triple_num = len(instance1) + len(attributes1) + len(relation1)
    gold_triple_num = len(instance2) + len(attributes2) + len(relation2)
    return smatch.compute_f(best_match_num, test_triple_num, gold_triple_num)
