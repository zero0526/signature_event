[INPUT] Danh sách N bài báo thuộc CÙNG 1 Cụm (Cluster)
   │
   ▼
[MODULE 1] FEATURE EXTRACTION (Chạy trên từng bài báo)
   │   (Dùng class VnCorePostprocessor của bạn)
   ├─ Bài 1 -> Lọc rác -> {Entities: [Đà Nẵng...], Actions: [vượt phải, bỏ chạy...]}
   ├─ Bài 2 -> Lọc rác -> {Entities: [tài xế...], Actions: [va chạm, rời khỏi...]}
   │
   ▼
[MODULE 2] CROSS-DOCUMENT AGGREGATION (Tổng hợp chéo)
   │   (Gộp các phrase giống nhau: "chiếc xe máy" + "xe máy" -> "xe máy")
   │   (Tính toán Document Frequency - DF)
   │
   ▼
[MODULE 3] SIGNATURE GENERATION (Sinh chữ ký)
   │   (Chỉ giữ lại những Entity/Action xuất hiện trên X% số bài báo)
   │
   ▼
[OUTPUT] Event Signature: { "core_entities": [...], "core_actions": [...] }