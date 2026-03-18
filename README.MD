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

[
   ('eo biển Hormuz', 0.12922869035613968),
   ('chở', 0.06591094457437634),
   ('Iran', 0.06202475617261197),
   ('Gargash', 0.0594878913528076),
   ('Mỹ', 0.058445048567084115),
   ('cho biết', 0.050650882565795786),
   ('UAE', 0.0504767616031059),
   ('giao dịch bằng', 0.04282062932504013),
   ('Israel', 0.040712597211432396),
   ('khiến', 0.034938429635439666),
   ('Trump', 0.034649097494092465),
   ('8 quốc gia', 0.03178255365190809),
   ('Trung Đông', 0.030902621440643395),
   ('ngại đưa', 0.02501642435466093),
   ('đảm bảo', 0.023098077214406633),
   ('AP', 0.022962818865378036),
   ('Tehran', 0.022962818865378036),
   ('là', 0.02287360486617241),
   ('qua', 0.019806658858397812),
   ('Trung Quốc', 0.01976076715095926),
   ('TPO', 0.01976076715095926),
   ('CNN', 0.019760767150959258),
   ('Liberia', 0.019563389095916737),
   ('cảng Mumbai Ấn Độ', 0.019563389095916733),
   ('bao gồm', 0.0194486312914124),
   ('Abu Dhabi', 0.016225214817879096),
   ('châu Á châu Âu', 0.016185893346755804),
   ('Mohammad Bagher Ghalibaf', 0.011917867235121619),
   ('nói', 0.00906200668924856)
]


[('eo biển Hormuz', 0.15216861650475869),
 ('Mỹ', 0.10972889108447557),
 ('chở dầu', 0.0662575729871176),
 ('Iran', 0.06237841869477039),
 ('UAE', 0.055153491799205166),
 ('Gargash', 0.050991775455447814),
 ('8 quốc gia', 0.05000614051675721),
 ('Trump', 0.048027878163494214),
 ('đảm bảo an ninh', 0.038078088671212956),
 ('đưa tàu chiến', 0.03336043271098777),
 ('khiến các nước ngại', 0.03336043271098776),
 ('ngại tàu chiến', 0.03336043271098776),
 ('Trung Đông', 0.02922220387957963),
 ('AP', 0.02736947774294505),
 ('Tehran', 0.02736947774294505),
 ('qua eo biển', 0.02509542926045418),
 ('châu Á châu Âu', 0.019945344203334464),
 ('cảng Mumbai Ấn Độ', 0.018468529442797824),
 ('Liberia', 0.018468529442797824),
 ('TPO', 0.017704067909511227),
 ('Trung Quốc', 0.017704067909511227),
 ('CNN', 0.017704067909511227),
 ('Israel', 0.016651125820687066),
 ('Mohammad Bagher Ghalibaf', 0.015921570018664233),
 ('Abu Dhabi', 0.015503936707058066)],