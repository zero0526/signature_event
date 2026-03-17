from copy import deepcopy
from typing import Dict, Any, List
class ExtractNER:
    def __init__(self,
                 merge_adjacent_same_label: bool = True,
                 expand_by_dep: bool = True,
                 merge_if_gap_connectors: bool = True,
                 keep_underscores: bool = False,
                 max_merge_gap: int = 1,
                 debug: bool = False):
        self.merge_adjacent_same_label = merge_adjacent_same_label
        self.expand_by_dep = expand_by_dep
        self.merge_if_gap_connectors = merge_if_gap_connectors
        self.keep_underscores = keep_underscores
        self.max_merge_gap = max_merge_gap
        self.debug = debug
        
    def process(self, raw: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Process text và trả về cấu trúc rõ ràng:
        - entities: Các thực thể định danh (NER)
        - events: Các cặp [Chủ thể] -> [Hành động] -> [Đối tượng]
        """
        out_entities =[]
        out_events =[]

        for sent_id, tokens in raw.items():
            tokens_copy = deepcopy(tokens)
            index_map = {t["index"]: t for t in tokens_copy}
            
            # 1. Extract NER Entities (giữ nguyên logic cũ của bạn)
            entities = self._merge_bio(tokens_copy)
            if self.expand_by_dep or self.merge_adjacent_same_label or self.merge_if_gap_connectors:
                entities = self._postprocess_entities(entities, tokens_copy, index_map)
            
            for e in entities:
                e["text"] = self._render_text(e["token_indices"], index_map)
                out_entities.append(e["text"])

            # 2. Extract Events (Hành động + Thực thể liên quan)
            events = self._extract_events(tokens_copy, index_map)
            out_events.extend(events)

        return {
            "ner_entities": list(set(out_entities)),
            "events": out_events
        }
