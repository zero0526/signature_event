from typing import List, Dict, Any, Tuple, Optional
import copy
from configs import configs
from utils import otsu_threshold, render_text
from entity_extractor import EntityExtractor
from event_extractor import EventExtractor
from quantity_extractor import QuantityExtractor
class VnCorePostprocessor:
    def __init__(self,
                 merge_adjacent_same_label: bool = True,
                 expand_by_dep: bool = True,
                 merge_if_gap_connectors: bool = True,
                 keep_underscores: bool = False,
                 max_merge_gap: int = 1,
                 debug: bool = False):
        self.keep_underscores = keep_underscores
        
        self.entity_extractor = EntityExtractor(
            merge_adjacent_same_label=merge_adjacent_same_label,
            expand_by_dep=expand_by_dep,
            merge_if_gap_connectors=merge_if_gap_connectors,
            max_merge_gap=max_merge_gap,
            debug=debug
        )
        self.event_extractor = EventExtractor(keep_underscores=keep_underscores)
        self.quantity_extractor = QuantityExtractor()

    def process(self, raw: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Process text và trả về cấu trúc rõ ràng:
        - entities: Các thực thể định danh (NER)
        - events: Các cặp [Chủ thể] -> [Hành động] -> [Đối tượng]
        """
        raw: Dict[int, List[Dict[str, Any]]]
        out_entities = []
        out_events = []
        
        quantity = self.quantity_extractor.extract(raw)
        
        for sent_id, tokens in raw.items():
            tokens_copy = copy.deepcopy(tokens)
            index_map = {t["index"]: t for t in tokens_copy}

            # 1. Extract NER Entities
            entities = self.entity_extractor.extract(tokens_copy, index_map)
            
            for e in entities:
                e["text"] = render_text(e["token_indices"], index_map, self.keep_underscores)
                pos_tags = tuple(t.get("posTag", "") for t in e["tokens"])
                ner_tags = tuple(t.get("nerLabel", "O") for t in e["tokens"])
                out_entities.append((sent_id, e["text"], pos_tags, ner_tags))

            # 2. Extract Events
            events = self.event_extractor.extract(tokens_copy, index_map)
            for e in events:
                out_events.append((sent_id, e))

        return {
            "ner_entities": list(set(out_entities)),
            "actions": self.event_extractor.filter_ostu(out_events),
            "quantities": quantity
        }

