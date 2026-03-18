from typing import List, Dict, Any, Tuple, Optional
import copy
from configs import configs
from utils import otsu_threshold, render_text

class EventExtractor:
    """Class extracted to handle extracting and analyzing events."""
    
    def __init__(self, keep_underscores: bool = False):
        self.keep_underscores = keep_underscores

    def extract(self, tokens: List[Dict[str, Any]], index_map: Dict[int, Dict[str, Any]]) -> List[Dict]:
        events = []
        visited_verbs = set()

        for t in tokens:
            idx = t["index"]
            pos = t.get("posTag", "")

            if pos in configs.VERB_POS and idx not in visited_verbs:
                vp_indices = self._expand_verb_phrase(idx, tokens, index_map)
                visited_verbs.update(vp_indices)

                action_text = render_text(vp_indices, index_map, self.keep_underscores)

                subject = self._find_dependent_phrase(idx, index_map, tokens, dep_types=["sub", "nsubj", "csubj"])
                object_ = self._find_dependent_phrase(idx, index_map, tokens, dep_types=["dob", "pob", "obj", "dobj"])

                if subject or object_ or len(vp_indices) > 1:
                    events.append({
                        "subject": subject if subject else "Unknown",
                        "action": action_text,
                        "object": object_ if object_ else "Unknown"
                    })

        return events

    def _expand_verb_phrase(self, head_idx, tokens, index_map):
        vp = set([head_idx])

        # ===== 1. expand qua dependency (chuẩn nhất) =====
        for t in tokens:
            if t.get("head") == head_idx:
                dep = t.get("depLabel", "").lower()
                pos = t.get("posTag", "")

                # giữ các modifier quan trọng
                if any(x in dep for x in ["aux", "adv", "neg", "amod", "compound"]):
                    vp.add(t["index"])

        # ===== 2. expand trái (rất hạn chế) =====
        for i in range(head_idx - 1, max(0, head_idx - 3), -1):
            if i not in index_map:
                break

            t = index_map[i]
            word = t["wordForm"].lower()

            # whitelist cụ thể
            if word in ["đã", "đang", "sẽ", "không", "chưa"]:
                vp.add(i)
            else:
                break

        # ===== 3. expand phải (cực kỳ chặt) =====
        for i in range(head_idx + 1, head_idx + 3):
            if i not in index_map:
                break

            t = index_map[i]
            pos = t.get("posTag", "")
            dep = t.get("depLabel", "").lower()

            # chỉ giữ nếu là verb phụ thuộc
            if pos in configs.VERB_POS and "compound" in dep:
                vp.add(i)
            else:
                break

        return sorted(vp)

    def _find_dependent_phrase(self, head_verb_idx: int, index_map: Dict[int, Dict], tokens: List[Dict], dep_types: List[str]) -> str:
        target_idx = None
        for t in tokens:
            if t.get("head") == head_verb_idx and any(d in t.get("depLabel", "").lower() for d in dep_types):
                target_idx = t["index"]
                break

        if not target_idx:
            if "sub" in dep_types[0]:
                for i in range(head_verb_idx - 1, max(0, head_verb_idx - 5), -1):
                    if i in index_map and index_map[i].get("posTag") in configs.NOUN_POS:
                        target_idx = i
                        break
            else:
                for i in range(head_verb_idx + 1, min(len(tokens)+1, head_verb_idx + 5)):
                    if i in index_map and index_map[i].get("posTag") in configs.NOUN_POS:
                        target_idx = i
                        break

        if target_idx:
            np_indices = [target_idx]
            for t in tokens:
                if t.get("head") == target_idx and t["index"] != target_idx:
                    if t.get("posTag") != "CH":
                        np_indices.append(t["index"])

            np_indices = sorted(list(set(np_indices)))
            return render_text(np_indices, index_map, self.keep_underscores)

        return ""

    @staticmethod
    def filter_ostu(out_events):
        action_cnt = {}
        obj_cnt = {}
        action_sents = {}

        for sent_id, event in out_events:
            action = event["action"]
            subj = event["subject"]
            obj = event["object"]

            # Gom thành phrase: action + object (nếu có)
            action_obj = f"{action} {obj}" if obj != "Unknown" else action

            # count action_obj
            action_cnt[action_obj] = action_cnt.get(action_obj, 0) + 1

            # map action_obj -> sent_id
            if action_obj not in action_sents:
                action_sents[action_obj] = []
            action_sents[action_obj].append(sent_id)

            # count obj/subj
            if subj != "Unknown":
                obj_cnt[subj] = obj_cnt.get(subj, 0) + 1
            if obj != "Unknown":
                obj_cnt[obj] = obj_cnt.get(obj, 0) + 1

        important_actions = otsu_threshold(action_cnt)

        # build result: (action_obj, score, sent_ids)
        results = []
        for act_obj, score in important_actions.items():
            results.append((act_obj, score, action_sents.get(act_obj, [])))

        return sorted(results, key=lambda x: x[1], reverse=True)