from typing import List, Dict, Any, Tuple, Optional
import copy
from configs import configs

class EntityExtractor:
    """Class extracted to handle extracting and post-processing NER Entities."""
    
    def __init__(self,
                 merge_adjacent_same_label: bool = True,
                 expand_by_dep: bool = True,
                 merge_if_gap_connectors: bool = True,
                 max_merge_gap: int = 1,
                 debug: bool = False):
        self.merge_adjacent_same_label = merge_adjacent_same_label
        self.expand_by_dep = expand_by_dep
        self.merge_if_gap_connectors = merge_if_gap_connectors
        self.max_merge_gap = max_merge_gap
        self.debug = debug

    def extract(self, tokens: List[Dict[str, Any]], index_map: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        entities = self._merge_bio(tokens)
        if self.expand_by_dep or self.merge_adjacent_same_label or self.merge_if_gap_connectors:
            entities = self._postprocess_entities(entities, tokens, index_map)
        return entities

    def _merge_bio(self, tokens):
        entities = []
        cur = None
        for t in tokens:
            ner = t.get("nerLabel", "O")
            idx = t["index"]

            if ner == "O":
                if cur:
                    entities.append(cur)
                    cur = None
                continue

            if ner.startswith("B-"):
                if cur:
                    entities.append(cur)
                cur = {"label": ner[2:], "token_indices": [idx]}

            elif ner.startswith("I-"):
                tag = ner[2:]
                if not cur or cur["label"] != tag:
                    if cur:
                        entities.append(cur)
                    cur = {"label": tag, "token_indices": [idx]}
                else:
                    cur["token_indices"].append(idx)

            else:
                entities.append({"label": ner, "token_indices": [idx]})

        if cur:
            entities.append(cur)
        return entities

    def _postprocess_entities(self, entities: List[Dict[str, Any]], tokens: List[Dict[str, Any]],
                              index_map: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        # sort by first index
        entities = sorted(entities, key=lambda e: e["token_indices"][0])

        # optional: expand entity by adjacency (POS)
        if self.expand_by_dep:
            entities = self._expand_by_pos(entities, index_map)

        # optional: merge adjacent same-label with very small gap
        if self.merge_adjacent_same_label:
            entities = self._merge_adjacent_same_label(entities, index_map)

        # optional: merge across very small connector tokens (dangerous; off by default)
        if self.merge_if_gap_connectors:
            entities = self._merge_across_connectors(entities, index_map, tokens)

        # reassign tokens properly
        for e in entities:
            e["token_indices"] = sorted(set(e["token_indices"]))
            e["tokens"] = [index_map[i] for i in e["token_indices"]]

        return entities

    def _expand_by_pos(self, entities: List[Dict[str, Any]],
                       index_map: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        entities_out = []
        all_entity_tokens = set()
        for e in entities:
            all_entity_tokens.update(e["token_indices"])

        for e in entities:
            e_indices = list(e["token_indices"])
            max_idx = max(e_indices)
            for delta in range(1, 4): # expand right up to 3 tokens
                cand_idx = max_idx + delta
                cand = index_map.get(cand_idx)
                if not cand:
                    break
                if cand_idx in all_entity_tokens:
                    break
                pos = cand.get("posTag", "")
                if pos in configs.NOUN_POS:
                    if self.debug:
                        print(f"expand: entity {e['label']} add token {cand['wordForm']} idx {cand_idx} pos {pos}")
                    e_indices.append(cand_idx)
                    all_entity_tokens.add(cand_idx)
                    max_idx = cand_idx
                    continue
                break
            e["token_indices"] = sorted(e_indices)
            entities_out.append(e)
        return entities_out

    def _merge_adjacent_same_label(self, entities: List[Dict[str, Any]],
                                   index_map: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not entities:
            return entities
        merged = []
        cur = dict(entities[0])
        for nxt in entities[1:]:
            gap = nxt["token_indices"][0] - cur["token_indices"][-1] - 1
            if nxt["label"] == cur["label"] and gap <= self.max_merge_gap:
                ok_gap = True
                for mid_idx in range(cur["token_indices"][-1] + 1, nxt["token_indices"][0]):
                    tok = index_map.get(mid_idx)
                    if tok is None:
                        ok_gap = False
                        break
                    if tok.get("posTag") not in configs.CONNECTOR_POS:
                        ok_gap = False
                        break
                if ok_gap:
                    if self.debug:
                        print(f"merge_adjacent: merge {cur['token_indices']} + {nxt['token_indices']} label {cur['label']}")
                    cur["token_indices"] = cur["token_indices"] + nxt["token_indices"]
                    cur["heads"] = (cur.get("heads", []) or []) + nxt.get("heads", [])
                    cur["depLabels"] = (cur.get("depLabels", []) or []) + nxt.get("depLabels", [])
                    cur["tokens"] = (cur.get("tokens", []) or []) + nxt.get("tokens", [])
                    continue
            merged.append(cur)
            cur = dict(nxt)
        merged.append(cur)
        return merged

    def _merge_across_connectors(self, entities: List[Dict[str, Any]],
                                 index_map: Dict[int, Dict[str, Any]],
                                 tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not entities:
            return entities
        merged = []
        cur = dict(entities[0])
        for nxt in entities[1:]:
            gap = nxt["token_indices"][0] - cur["token_indices"][-1] - 1
            if nxt["label"] == cur["label"] and gap <= 3:
                mids = [index_map[i] for i in range(cur["token_indices"][-1] + 1, nxt["token_indices"][0])]
                if all(m.get("posTag") in configs.CONNECTOR_POS for m in mids):
                    if self.debug:
                        print(f"merge_connectors: merging across {len(mids)} connector tokens")
                    cur["token_indices"] = cur["token_indices"] + nxt["token_indices"]
                    cur["tokens"] = cur.get("tokens", []) + nxt.get("tokens", [])
                    continue
            merged.append(cur)
            cur = dict(nxt)
        merged.append(cur)
        return merged