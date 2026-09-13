# Kế hoạch thí nghiệm H/T

## Câu hỏi nghiên cứu

Với cùng tokenizer, corpus, token budget và pipeline SFT, candidate Mamba-2/global GQA
có giữ được khả năng chọn tool, sử dụng bằng chứng và exact recall so với pure GQA ở
ngân sách khoảng 373M tham số, trong khi cải thiện memory/latency thực tế hay không?
Không lấy chiến thắng ở scale khác làm bằng chứng kết luận cho tiny model này.

## Trình tự

**Kiểm tra implementation:** chạy toàn bộ CPU/BPE/CUDA tests; so parity Mamba reference
với upstream; kiểm tra backward BF16 và cache; chạy overfit một batch; đối chiếu train
resume và xem vài chục normalized/tokenized examples bằng tay.

**Pilot 59M:** ngân sách dự kiến 50–200 triệu token để kiểm tra xu hướng loss và dữ liệu;
đây là mốc kỹ thuật đề xuất, không phải ngưỡng chất lượng được chứng minh. Chạy thêm
pure Transformer pilot phù hợp ngân sách nếu muốn so chất lượng ở quy mô này.

**So sánh chính:** dùng `hybrid_373m.yaml` và `transformer_373m.yaml`. Vòng đầu có thể
đặt 1–3 tỷ predicted tokens tùy tài nguyên, rồi mở rộng 5–10 tỷ khi dữ liệu/metric hợp lý.
Đó là ngân sách thí nghiệm, không phải lời hứa đạt chất lượng với số token đó. Không so
một model đã SFT với một model chỉ pretrain. Báo số lần lặp dữ liệu, không chỉ tokens_seen.

**Context curriculum:** 2K rồi 4K/8K với dữ liệu có độ dài thật. Giữ RoPE theta nhất quán;
đổi context training qua --init/new-stage, không reset optimizer rồi gọi exact resume.
Không ngoại suy kết quả 8K thành 32K nếu chưa có eval tương ứng.

## Metric đã có trong code

- Validation loss/PPL: đúng theo target mask của mỗi giai đoạn.
- Conditional tool-plan JSON parsing, tool name exact match, name+arguments exact match.
- Conditional final answer string exact match, citation ID precision/recall.
- Random-weight prefill/decode timing và peak allocated CUDA memory.
- Agent CLI thực thi tool loop có trace; chưa phải benchmark vòng kín tổng hợp.

## Metric cần triển khai trước kết luận nghiên cứu

Associative retrieval và needle/value exact recall với distractors và độ dài tăng dần;
math accuracy trên đề độc lập với symbolic/numeric equivalence đáng tin cậy;
citation entailment thay vì chỉ ID tồn tại; RAG robustness với evidence thiếu/mâu thuẫn;
closed-loop tool execution success và tool failure recovery; latency phân vị trên workload
thực, số seed và khoảng bất định; FLOPs/energy nếu đó là tiêu chí tối ưu.

Không tự đặt các metric chưa có thành số trong báo cáo. Không gọi text exact match là
MATH accuracy hay citation ID precision là faithfulness. Triển khai task-specific
verifier hoặc human review cho claim quan trọng.

## Ablation

Sau khi baseline ổn định, so `MMMA`, `MA`, pure `A`, và thêm vị trí attention cuối.
Điều chỉnh FFN để ngân sách tham số gần nhau; báo chính xác chênh lệch. Thay một biến
mỗi lần. No-tool / call-only / complete-tool-loop / RAG evidence / missing-evidence
là những ablation dữ liệu hữu ích, tách biệt với ablation kiến trúc.

Tỷ lệ 3:1, FFN size, RoPE theta và mix weights hiện là lựa chọn thử nghiệm. Thất bại
ở recall hoặc tool arguments là lý do xem lại thiết kế/dữ liệu, không chỉ tăng token
hay tuyên bố model sẽ tự học khi đủ lâu.
