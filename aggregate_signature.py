# aggregate_signature.py
from collections import Counter
from typing import List, Dict

class EventSignatureAggregator:
    def __init__(self, min_support: float = 0.3):
        # min_support = 0.3 nghĩa là từ khóa phải xuất hiện ở ít nhất 30% số bài báo
        self.min_support = min_support

    def generate_signature(self, list_of_article_features: List[Dict[str, List[str]]]):
        total_docs = len(list_of_article_features)
        
        # Nếu cụm chỉ có 1-2 bài, hạ ngưỡng xuống để không bị rỗng
        actual_support = self.min_support if total_docs >= 3 else 0.0

        entity_counter = Counter()
        action_counter = Counter()

        # 1. Đếm Document Frequency (DF)
        for doc_features in list_of_article_features:
            # Dùng set để mỗi bài báo chỉ tính 1 lần cho 1 từ (dù nó lặp lại 10 lần trong bài)
            unique_entities = set([e.lower().strip() for e in doc_features["entities"]])
            unique_actions = set([a.lower().strip() for a in doc_features["actions"]])
            
            entity_counter.update(unique_entities)
            action_counter.update(unique_actions)

        # 2. Gộp các cụm từ bao hàm nhau (VD: "chiếc xe máy" và "xe máy" -> Gộp vào "xe máy")
        entity_counter = self._merge_similar_phrases(entity_counter)
        action_counter = self._merge_similar_phrases(action_counter)

        # 3. Lọc theo ngưỡng và sắp xếp
        core_entities =[
            phrase for phrase, count in entity_counter.items() 
            if (count / total_docs) >= actual_support
        ]
        core_actions =[
            phrase for phrase, count in action_counter.items() 
            if (count / total_docs) >= actual_support
        ]

        # Sắp xếp theo mức độ phổ biến
        core_entities = sorted(core_entities, key=lambda x: entity_counter[x], reverse=True)
        core_actions = sorted(core_actions, key=lambda x: action_counter[x], reverse=True)

        return {
            "total_articles": total_docs,
            "core_entities": core_entities[:10], # Top 10 đặc trưng nhất
            "core_actions": core_actions[:10]
        }

    def _merge_similar_phrases(self, phrase_counter: Counter) -> Counter:
        """Thuật toán gộp các cụm từ tương đồng (Substring matching)"""
        merged = Counter()
        sorted_phrases = sorted(phrase_counter.keys(), key=len) # Ngắn đến dài
        
        for phrase in sorted_phrases:
            count = phrase_counter[phrase]
            is_merged = False
            
            for kept_phrase in list(merged.keys()):
                # Nếu cụm này là tập con của cụm kia hoặc ngược lại
                if kept_phrase in phrase or phrase in kept_phrase:
                    target = kept_phrase if len(kept_phrase) < len(phrase) else phrase
                    if target != kept_phrase:
                        merged[target] = merged.pop(kept_phrase) + count
                    else:
                        merged[kept_phrase] += count
                    is_merged = True
                    break
                    
            if not is_merged:
                merged[phrase] = count
                
        return merged