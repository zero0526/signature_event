from typing import List, Dict, Any
from configs import configs

class EventFeatureExtractor:
    def __init__(self, 
                keep_underscores= False):
        self.keep_underscores= keep_underscores
    
    def _extract_events(self, tokens: List[Dict[str, Any]], index_map: Dict[int, Dict[str, Any]]) -> List[Dict]:
        """
        Trích xuất cụm động từ và tìm Chủ ngữ (Subject), Tân ngữ (Object) của nó dựa vào Dependency.
        """
        events =[]
        visited_verbs = set()

        for t in tokens:
            idx = t["index"]
            pos = t.get("posTag", "")

            # Nếu là Động từ và chưa được duyệt
            if pos in configs.VERB_POS and idx not in visited_verbs:
                # 2.1 Mở rộng để lấy Cụm Hành Động (Verb Phrase)
                vp_indices = self._expand_verb_phrase(idx, index_map)
                visited_verbs.update(vp_indices)

                action_text = self._render_text(vp_indices, index_map)

                # 2.2 Tìm Chủ ngữ (Subject) và Tân ngữ (Object) liên kết với Động từ chính (idx)
                subject = self._find_dependent_phrase(idx, index_map, tokens, dep_types=["sub", "nsubj", "csubj"])
                object_ = self._find_dependent_phrase(idx, index_map, tokens, dep_types=["dob", "pob", "obj", "dobj"])

                # Chỉ lưu lại những hành động có ý nghĩa (có subject hoặc object, hoặc là cụm động từ dài)
                if subject or object_ or len(vp_indices) > 1:
                    events.append({
                        "subject": subject if subject else "Unknown",
                        "action": action_text,
                        "object": object_ if object_ else "Unknown"
                    })

        return events
    
    def _find_dependent_phrase(self, head_verb_idx: int, index_map: Dict[int, Dict], tokens: List[Dict], dep_types: List[str]) -> str:
        """
        Tìm cụm danh từ đóng vai trò Subject hoặc Object trỏ đến động từ chính.
        """
        target_idx = None
        # Tìm token nào có head trỏ tới động từ và có nhãn dependency phù hợp (sub hoặc dob)
        for t in tokens:
            if t.get("head") == head_verb_idx and any(d in t.get("depLabel", "").lower() for d in dep_types):
                target_idx = t["index"]
                break

        if not target_idx:
            # Fallback heuristic: Nếu không bắt được qua dependency, thử tìm Danh từ gần nhất
            if "sub" in dep_types[0]: # Tìm ngược lên cho subject
                for i in range(head_verb_idx - 1, max(0, head_verb_idx - 5), -1):
                    if i in index_map and index_map[i].get("posTag") in configs.NOUN_POS:
                        target_idx = i
                        break
            else: # Tìm xuôi xuống cho object
                for i in range(head_verb_idx + 1, min(len(tokens)+1, head_verb_idx + 5)):
                    if i in index_map and index_map[i].get("posTag") in configs.NOUN_POS:
                        target_idx = i
                        break

        if target_idx:
            # Mở rộng để lấy cả Cụm danh từ (Ví dụ: "xe" -> "xe ô tô này")
            np_indices = [target_idx]
            # Quét các từ bổ nghĩa xung quanh có head trỏ về target_idx
            for t in tokens:
                if t.get("head") == target_idx and t["index"] != target_idx:
                    # Bỏ qua dấu câu
                    if t.get("posTag") != "CH":
                        np_indices.append(t["index"])

            np_indices = sorted(list(set(np_indices)))
            return self._render_text(np_indices, index_map)

        return ""
    
    def _expand_verb_phrase(self, head_idx: int,index_map: Dict[int, Dict[str, Any]]) -> List[int]:
        """
        Gộp các Phó từ (R), Tính từ (A), Giới từ (E) đứng cạnh Động từ để tạo thành cụm có nghĩa.
        Ví dụ: cố_tình (R) + vượt (V) + phải (A) -> cố tình vượt phải
        """
        vp_indices = [head_idx]

        # Mở rộng sang trái (Tìm phó từ bổ nghĩa: đã, đang, sẽ, cố tình...)
        curr_idx = head_idx - 1
        while curr_idx in index_map:
            t = index_map[curr_idx]
            if t.get("posTag") in configs.ADV_POS:  # R: Phó từ
                vp_indices.insert(0, curr_idx)
                curr_idx -= 1
            else:
                break
            
    def _render_text(self, token_indices: List[int], index_map: Dict[int, Dict]) -> str:
        words = [index_map[i]["wordForm"] for i in token_indices]
        if not self.keep_underscores:
            words = [w.replace("_", " ") for w in words]
        return " ".join(words)