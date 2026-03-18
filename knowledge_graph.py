from collections import defaultdict
import re
from igraph import Graph


class KeywordGraph:

    def __init__(self, stopword_path: str, sim_threshold=100):
        self.sim_threshold = sim_threshold  # fuzz ratio
        self.stop_word: set= self._load_stopwords(stopword_path)
    
    def _load_stopwords(self, stopword_path):
        try:
            with open(stopword_path, "r", encoding="utf-8") as f:
                stop_words = f.read().split("\n")
            print(f"Loaded {len(stop_words)} stopwords.")
            return set(stop_words)
        except FileNotFoundError:
            print(f"Error: The file not found at {stopword_path}")
            return {}
        except Exception as e:
            print(f"An error occurred: {e}")
            return {}

    # ========= NORMALIZE =========
    def normalize(self, text):
        text = text.lower()
        text = text.replace("_", " ")
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    # ========= SIMILARITY =========
    def is_similar(self, a, b):
        a_norm = self.normalize(a)
        b_norm = self.normalize(b)

        # substring ưu tiên
        if a_norm in b_norm or b_norm in a_norm:
            return True

        return False

    # ========= ITEM RESOLUTION =========
    def resolve_items(self, items, item_type="entity"):
        groups = []

        if item_type == "action":
            # items: [(text, score, sent_ids), ...]
            for text, score, sids in items:
                matched = False
                for g in groups:
                    if self.is_similar(text, g["rep"]):
                        g["items"].append((sids, text))
                        matched = True
                        break

                if not matched:
                    groups.append({"rep": text, "items": [(sids, text)]})
        elif item_type == "entity":
            # items: [(sid, text, pos_tags, ner_tags), ...]
            for sid, text, pos_tags, ner_tags in items:
                matched = False
                for g in groups:
                    if self.is_similar(text, g["rep"]):
                        g["items"].append((sid, text, pos_tags, ner_tags))
                        matched = True
                        break

                if not matched:
                    groups.append({"rep": text, "items": [(sid, text, pos_tags, ner_tags)]})
        else:
            # items: [(sid, text), ...] (quantity)
            for sid, text in items:
                matched = False
                for g in groups:
                    if self.is_similar(text, g["rep"]):
                        g["items"].append((sid, text))
                        matched = True
                        break

                if not matched:
                    groups.append({"rep": text, "items": [(sid, text)]})

        canonical = []
        mention_map = {}

        for g in groups:
            if item_type == "action":
                texts = [t for _, t in g["items"]]
                name = min(texts, key=len)
                sent_ids = []
                for sids, _ in g["items"]:
                    sent_ids.extend(sids)
                sent_ids = list(set(sent_ids))
            elif item_type == "entity":
                texts = [t for _, t, _, _ in g["items"]]
                sorted_entities = sorted(g["items"], key=lambda x: len(x[1]))
                best_entity = sorted_entities[0] # Ưu tiên ngắn nhất
                
                # Check theo rule ưu tiên
                for cand in sorted_entities[1:]:
                    best_tok_count = len(best_entity[2])
                    cand_tok_count = len(cand[2])
                    if cand_tok_count > best_tok_count:
                        extra_pos = cand[2][best_tok_count:]
                        extra_ner = cand[3][best_tok_count:]
                        
                        has_important = False
                        for p, n in zip(extra_pos, extra_ner):
                            if p in {"Ny", "Nd", "Nu", "M"} or n.startswith("B-") or n.startswith("I-"):
                                has_important = True
                                break
                        
                        if has_important:
                            best_entity = cand # Đổi ưu tiên sang tên dài
                    elif cand_tok_count == best_tok_count == 1 and len(cand[1]) > len(best_entity[1]):
                        p = cand[2][0]
                        n = cand[3][0]
                        if p in {"Ny", "Nd", "Nu", "M"} or n.startswith("B-") or n.startswith("I-"):
                            best_entity = cand
                
                name = best_entity[1]
                sent_ids = list(set(s for s, _, _, _ in g["items"]))
            else:
                texts = [t for _, t in g["items"]]
                name = min(texts, key=len)
                sent_ids = list(set(s for s, _ in g["items"]))

            canonical.append({
                "name": name,
                "mentions": texts,
                "sent_ids": sent_ids
            })

            if item_type == "entity":
                for _, t, _, _ in g["items"]:
                    mention_map[t] = name
            else:
                for _, t in g["items"]:
                    mention_map[t] = name

        return canonical, mention_map

    # ========= BUILD GRAPH =========

    def build_graph(self, data):
        ner_entities = data.get("ner_entities", [])
        actions = data.get("actions", [])
        quantities = data.get("quantities", [])

        canonical_entities, entity_map = self.resolve_items(ner_entities, "entity")
        canonical_actions, action_map = self.resolve_items(actions, "action")
        canonical_quantities, quantity_map = self.resolve_items(quantities, "quantity")

        # ===== group theo sentence =====
        entities_by_sent = defaultdict(set)
        for sid, e_text, pos_tags, ner_tags in ner_entities:
            entities_by_sent[sid].add(entity_map.get(e_text, e_text))

        quantities_by_sent = defaultdict(set)
        for sid, q in quantities:
            if q not in self.stop_word:
                quantities_by_sent[sid].add(quantity_map.get(q, q))

        actions_by_sent = defaultdict(set)
        for raw_action, score, sent_ids in actions:
            action = raw_action.replace("_", " ").lower()
            if action in self.stop_word:
                continue
            resolved_action = action_map.get(raw_action, raw_action)
            for sid in sent_ids:
                actions_by_sent[sid].add(resolved_action)

        # ===== 1. BUILD NODE LIST (QUAN TRỌNG NHẤT) =====
        nodes = set()

        # entities
        for ents in entities_by_sent.values():
            nodes.update(ents)

        # actions
        for acts in actions_by_sent.values():
            nodes.update(acts)

        # quantities
        for qs in quantities_by_sent.values():
            nodes.update(qs)

        nodes = list(nodes)
        node_index = {n: i for i, n in enumerate(nodes)}

        # ===== 2. BUILD EDGE =====
        edge_weights = defaultdict(float)

        def add_edge(a, b, w=1.0):
            if a == b:
                return
            key = tuple(sorted([a, b]))
            edge_weights[key] += w

        # ===== entity ↔ action =====
        for sid in entities_by_sent:
            ents = entities_by_sent[sid]
            acts = actions_by_sent.get(sid, [])

            for e in ents:
                for a in acts:
                    add_edge(e, a, w=3.0)  # weight cao hơn

        # ===== entity ↔ quantity =====
        for sid in entities_by_sent:
            ents = entities_by_sent[sid]
            qs = quantities_by_sent.get(sid, [])

            for e in ents:
                for q in qs:
                    add_edge(e, q, w=2.0)

        # ===== entity ↔ entity =====
        for sid, ents in entities_by_sent.items():
            ents = list(ents)
            if len(ents) <= 5:
                for i in range(len(ents)):
                    for j in range(i + 1, len(ents)):
                        add_edge(ents[i], ents[j], w=1.0)

        # ===== 3. BUILD GRAPH =====
        g = Graph()
        g.add_vertices(len(nodes))
        g.vs["name"] = nodes

        if edge_weights:
            edges = [(node_index[a], node_index[b]) for a, b in edge_weights]
            weights = list(edge_weights.values())

            g.add_edges(edges)
            g.es["weight"] = weights
        else:
            g.es["weight"] = []

        # ===== 4. HANDLE ISOLATED NODE =====
        degrees = g.degree()
        for i, deg in enumerate(degrees):
            if deg == 0:
                # self-loop nhẹ để không bị chết pagerank
                g.add_edge(i, i)
                g.es[-1]["weight"] = 0.01

        return g, canonical_entities

    # ========= TEXTRANK =========
    def textrank(self, graph):
        scores = graph.pagerank(weights=graph.es["weight"])
        return {
            graph.vs[i]["name"]: scores[i]
            for i in range(len(scores))
        }

    # ========= MULTI-DOCUMENT MERGING =========
    def resolve_nodes(self, nodes):
        """Gom nhóm các node (từ nhiều graph) nếu chúng giống/bao trùm nhau."""
        groups = []
        for text in nodes:
            matched = False
            for g in groups:
                if self.is_similar(text, g["rep"]):
                    g["items"].append(text)
                    matched = True
                    break
            if not matched:
                groups.append({"rep": text, "items": [text]})
        
        node_map = {}
        for g in groups:
            # Ưu tiên lấy node ngắn nhất (keyword cốt lõi) làm đại diện
            name = min(g["items"], key=len)
            for t in g["items"]:
                node_map[t] = name
        return node_map

    def merge_graphs(self, graphs):
        """Merge danh sách các Graph lại với nhau dựa trên sự bao trùm của keyword."""
        all_nodes = set()
        raw_edges = []
        
        # 1. Thu thập toàn bộ node và edge từ các graph
        for g in graphs:
            names = g.vs["name"]
            all_nodes.update(names)
            for edge in g.es:
                u_name = names[edge.source]
                v_name = names[edge.target]
                weight = edge["weight"]
                raw_edges.append((u_name, v_name, weight))
                
        # 2. Gom cụm các node giống nhau
        node_map = self.resolve_nodes(list(all_nodes))
        
        # 3. Tính toán lại Nodes và Edges
        merged_nodes = list(set(node_map.values()))
        node_index = {n: i for i, n in enumerate(merged_nodes)}
        
        edge_weights = defaultdict(float)
        for u, v, w in raw_edges:
            cu = node_map[u]
            cv = node_map[v]
            if cu == cv:
                continue
            key = tuple(sorted([cu, cv]))
            edge_weights[key] += w  # Cộng dồn weight
            
        # 4. Khởi tạo merged graph
        merged_g = Graph()
        merged_g.add_vertices(len(merged_nodes))
        merged_g.vs["name"] = merged_nodes
        
        if edge_weights:
            edges = [(node_index[a], node_index[b]) for a, b in edge_weights]
            weights = list(edge_weights.values())
            merged_g.add_edges(edges)
            merged_g.es["weight"] = weights
        else:
            merged_g.es["weight"] = []
            
        # Tự liên kết các node cô lập nhẹ
        degrees = merged_g.degree()
        for i, deg in enumerate(degrees):
            if deg == 0:
                merged_g.add_edge(i, i)
                merged_g.es[-1]["weight"] = 0.01

        return merged_g

    # ========= MAIN =========
    def run(self, data):
        g, entities = self.build_graph(data)
        scores = self.textrank(g)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "entities": entities,
            "ranked_keywords": ranked,
            "graph": g
        }

    def run_multiple(self, data_list):
        """Chạy tổng hợp trên nhiều bài viết (list các data map)."""
        graphs = []
        for data in data_list:
            g, _ = self.build_graph(data)
            graphs.append(g)

        # Trộn tất cả graph
        merged_g = self.merge_graphs(graphs)
        
        # Chạy TextRank một lần cuối trên graph tổng
        scores = self.textrank(merged_g)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "ranked_keywords": ranked,
            "graph": merged_g
        }