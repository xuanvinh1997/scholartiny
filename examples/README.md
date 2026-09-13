# Ví dụ nhỏ, không phải dữ liệu huấn luyện hoàn chỉnh

`train.jsonl` là hai đoạn văn bản tự soạn để minh họa schema pretrain. `academic_sft.jsonl` gồm 14 bài tập tự sinh (bảy nhóm nhiệm vụ, tiếng Việt và tiếng Anh) dùng cùng generator với repo. Báo cáo thí nghiệm trong ví dụ là hư cấu.

Các file này không được lấy từ public dataset. Chúng không đủ để huấn luyện tokenizer 32K hoặc model học thuật. Không dùng làm benchmark chất lượng. Lệnh `python -m scripts.smoke_test --out .work/demo` tự tạo bộ fixture thích hợp cho toàn pipeline.
