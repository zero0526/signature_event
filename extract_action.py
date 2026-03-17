# extract_features.py
from vncore_postprocessor import VnCorePostprocessor # Import class của bạn

# Từ điển rác báo chí
_STOP_VERBS = {"cho biết", "thông tin", "tiếp nhận", "xác minh", "điều tra", "lan truyền", "đăng tải", "bức xúc"}
_STOP_NOUNS = {"cơ quan chức năng", "mạng xã hội", "dư luận", "thông tin", "vụ việc", "đoạn clip", "đơn vị"}

class ArticleFeatureExtractor:
    def __init__(self, vncore_model):
        self.pp = VnCorePostprocessor(vncore_model)

    def extract(self, text: str):
        # Chạy logic của bạn
        raw_result = self.pp.process(text) 
        
        clean_entities = set()
        clean_actions = set()

        # 1. Làm sạch Thực thể (NER & Objects)
        for ner in raw_result["ner_entities"]:
            ner_lower = ner.lower()
            if not any(stop in ner_lower for stop in _STOP_NOUNS) and len(ner.split()) > 1:
                clean_entities.add(ner)

        # 2. Làm sạch Hành động từ Events
        for event in raw_result["events"]:
            action = event["action"].lower()
            subject = event["subject"].lower()
            object_ = event["object"].lower()

            # Bỏ qua nếu hành động là từ báo cáo (VD: "cho biết")
            if not any(stop in action for stop in _STOP_VERBS):
                # Thêm action vào tập hợp
                clean_actions.add(action)
                
                # Trích xuất thêm Object từ cụm sự kiện nếu nó không phải là rác
                if object_ != "unknown" and not any(stop in object_ for stop in _STOP_NOUNS):
                    clean_entities.add(event["object"])
                if subject != "unknown" and not any(stop in subject for stop in _STOP_NOUNS):
                    clean_entities.add(event["subject"])

        return {
            "entities": list(clean_entities),
            "actions": list(clean_actions)
        }