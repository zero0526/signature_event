from typing import List, Dict, Any
from configs import configs
import re 

class QuantityExtractor:

    VALID_DEPS = {
        "nmod", "amod", "compound", "pob", "loc"
    }

    VALID_POS = {"N", "Np", "Nc", "M", "Nu"}

    @staticmethod
    def is_valid_number(token):
        text = token["wordForm"]

        # match số kiểu VN + %
        return bool(re.match(r"^\d+([.,]\d+)*%?$", text))

    @staticmethod
    def build_index(sent):
        """Tạo map head -> children"""
        children = {}
        for tok in sent:
            head = tok["head"]
            children.setdefault(head, []).append(tok)
        return children

    @staticmethod
    def expand(node, children_map, visited):
        """DFS expand theo dependency"""
        results = []

        for child in children_map.get(node["index"], []):
            if child["index"] in visited:
                continue
            
            if (child["depLabel"] in QuantityExtractor.VALID_DEPS and
                child["posTag"] in QuantityExtractor.VALID_POS):

                visited.add(child["index"])
                results.append(child)

                # recursive expand
                results.extend(
                    QuantityExtractor.expand(child, children_map, visited)
                )

        return results

    @staticmethod
    def extract(annotations: Dict[int, List[Dict[str, Any]]]) -> List[str]:
        results = []

        for index, sent in annotations.items():
            children_map = QuantityExtractor.build_index(sent)

            for token in sent:
                # 1. anchor = số
                if token.get("posTag") != "M":
                    continue
                if not QuantityExtractor.is_valid_number(token):
                    continue
                visited = set()
                visited.add(token["index"])

                phrase_tokens = [token]

                # 2. lấy head (thường là noun chính)
                head_idx = token["head"]
                head_token = next((t for t in sent if t["index"] == head_idx), None)

                if head_token and head_token["posTag"] in QuantityExtractor.VALID_POS:
                    phrase_tokens.append(head_token)
                    visited.add(head_token["index"])

                    for t in sent:
                        if t["head"] == head_idx and t["index"] not in visited:
                            if t["posTag"] in QuantityExtractor.VALID_POS:
                                # filter CH, chỉ giữ %
                                if t["posTag"] == "CH" and t["wordForm"] != "%":
                                    continue

                                phrase_tokens.append(t)
                                visited.add(t["index"])

                    # 3. expand từ head (DFS xuống dưới)
                    expanded = QuantityExtractor.expand(head_token, children_map, visited)
                    phrase_tokens.extend(expanded)

                # 4. sort theo index để đúng thứ tự câu
                phrase_tokens = sorted(phrase_tokens, key=lambda x: x["index"])

                phrase = " ".join(t["wordForm"].replace("_", " ") for t in phrase_tokens)

                if len(phrase_tokens) >= 2:
                    results.append((index,phrase))

        return results