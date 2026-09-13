# Data policy, provenance và giới hạn

1. Nguồn công khai không đồng nghĩa license tùy ý. Không tải nguồn gated khi chưa được
   tài khoản cấp quyền. Collector lấy HF_TOKEN từ môi trường và không lưu token trong DB.
   Cờ acknowledge là xác nhận người vận hành đã review, không tự tạo quyền pháp lý.
2. Snapshot resolve commit SHA, lưu card và nhãn license, cấu hình adapter, seed,
   số mẫu giữ/loại. Trường URL/title/ID chỉ có khi upstream cung cấp; không dựng nguồn giả.
3. Normalize NFC/newline, không lower-case hay xóa LaTeX trong text training. Group-key
   chuẩn hóa whitespace riêng để chia split và dò overlap. Exact dedup nội dung toàn cục;
   duplicate provenance giữ ở SQLite, primary record có nguồn xuất hiện đầu tiên.
4. Nhiều lời giải một bài toán dùng cùng group ID. RAG warmup kế thừa split gốc. Tách
   train/val/test trước tokenization; không train tokenizer bằng val/test.
5. `--deny-file` phát hiện exact/13-word overlap, không phát hiện chắc chắn paraphrase,
   dịch ngôn ngữ, bản chụp biểu thức hoặc template-equivalent math problems. Cần audit
   benchmark độc lập; không công bố “contamination-free” chỉ dựa vào filter này.
6. Các adapter bắt lỗi schema phổ biến, filter length và NUL. Đây không phải bộ lọc chất
   lượng học thuật, lỗi toán, dữ liệu cá nhân hoặc nội dung độc hại hoàn chỉnh.
7. Dataset tổng hợp có thể sai. OpenMath augmented expected_answer là majority-voting
   answer theo card, không phải ground truth hình thức. Chỉ tin kết quả công cụ đối với
   bài có giả thiết/biểu thức đã xác định đúng và xác minh độc lập khi cần.
8. Giữ train và eval benchmark ngoài pipeline collector riêng biệt. Không crawl toàn bộ
   arXiv PDF vì “public”; license cụ thể, parser công thức và quyền tái phân phối cần thiết
   kế riêng. Repo không chứa universal arXiv scraper hoặc tài liệu paywall.
9. Không đưa Python-Edu vào registry chạy ngay: subset đó không đơn giản là cột code text;
   cần xử lý nội dung Software Heritage và license file/repository đúng quy trình upstream.
10. Dataset license không bị thay bằng MIT của repository. RAG warmup kế thừa provenance
    tài liệu; MIT chỉ áp dụng code mới và các fixture hoàn toàn do generator của repo tạo.

## Reproducibility

Giữ `collection.sqlite`, toàn bộ `normalized/*/manifest.json`, các file tokenized,
`tokenizer.json` và fingerprint, source registry đã dùng, run.json/metrics/checkpoint,
package freeze sau khi GPU tests qua. Đổi code normalizer hoặc filters phải bump signature
version và tạo dataset artifact mới. Không giữ DB cũ rồi tuyên bố dữ liệu đã được lọc lại.

SQLite transaction giúp resume những record đã commit; export dùng file tạm rồi replace.
Collector single-process; không cho nhiều tiến trình cùng export một source vào cùng đường dẫn.
`--limit` là tổng số mẫu unique thuộc source đầu tiên trong ledger; nguồn nhiều trùng lặp có
thể đạt ít mẫu hơn khi chạm max_seen. Kiểm tra rejected statistics, không bỏ qua manifest.

SFT quá dài bị drop nguyên hội thoại; tăng context và re-prepare ở thư mục mới khi cần.
Tỷ lệ sampling SFT theo example: hội thoại dài góp nhiều target token hơn. Đánh giá cả
mix-weighted validation và từng nguồn riêng, không chỉ validation synthetic.
