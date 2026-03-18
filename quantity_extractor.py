from typing import List, Dict, Any, Tuple
from configs import configs
import re 

class QuantityExtractor:

    VALID_DEPS = set() # Not used anymore

    VALID_POS = {"N", "Np", "Nc", "M", "Nu"}

    @staticmethod
    def is_valid_number(token):
        text = token["wordForm"]
        # match số kiểu VN + %
        return bool(re.match(r"^\d+([.,]\d+)*%?$", text))

    @staticmethod
    def extract(annotations: Dict[int, List[Dict[str, Any]]]) -> List[Tuple[int, str]]:
        results = []

        for index, sent in annotations.items():
            for i, token in enumerate(sent):
                # 1. anchor = số
                if token.get("posTag") != "M":
                    continue
                if not QuantityExtractor.is_valid_number(token):
                    continue

                phrase_tokens = [token]
                
                # 2. expand left (nouns, other numbers)
                for j in range(i - 1, max(-1, i - 4), -1):
                    t = sent[j]
                    if t.get("posTag") in QuantityExtractor.VALID_POS:
                        phrase_tokens.append(t)
                    else:
                        break
                        
                # 3. expand right (units, nouns, percentage)
                for j in range(i + 1, min(len(sent), i + 4)):
                    t = sent[j]
                    if t.get("posTag") in QuantityExtractor.VALID_POS:
                        phrase_tokens.append(t)
                    elif t.get("posTag") == "CH" and t["wordForm"] == "%":
                        phrase_tokens.append(t)
                    else:
                        break

                # 4. sort theo index để đúng thứ tự câu
                phrase_tokens = sorted(phrase_tokens, key=lambda x: x["index"])

                phrase = " ".join(t["wordForm"].replace("_", " ") for t in phrase_tokens)

                if len(phrase_tokens) >= 2:
                    results.append((index, phrase))

        return list(set(results))