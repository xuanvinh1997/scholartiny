# ScholarTiny — Hybrid Mamba-2 + Global GQA

Repo nghiên cứu theo cách tổ chức dễ đọc của MiniMind: model PyTorch riêng, script thu thập dữ liệu, tokenizer, pretrain, full SFT, suy luận, tool calling, RAG và kiểm thử. Đây là **mã nguồn mới**, không phải fork hay bản phát hành chính thức của MiniMind. Chưa kèm trọng số học thuật đã huấn luyện.

Mục tiêu: một LLM nhỏ đóng vai trò trợ lý học thuật và bộ điều phối công cụ. Kiến thức và bằng chứng có thể nằm trong RAG; tính toán được giao cho công cụ; model học chọn thao tác, sử dụng quan sát và trả lời có căn cứ. Không mặc định rằng đổi kiến trúc sẽ tạo ra khả năng toán học tốt.

## 1. Kiến trúc được triển khai

```text
Token IDs → Embedding
              │
              ├─ RMSNorm → Mamba-2    → residual → RMSNorm → SwiGLU → residual
              ├─ RMSNorm → Mamba-2    → residual → RMSNorm → SwiGLU → residual
              ├─ RMSNorm → Mamba-2    → residual → RMSNorm → SwiGLU → residual
              └─ RMSNorm → Global GQA → residual → RMSNorm → SwiGLU → residual
                        lặp 6 lần
              │
              └─ RMSNorm → tied LM head
```

Global GQA có RoPE và Q/K RMSNorm theo từng head. Attention là causal trên toàn prefix, không phải sliding window. Tất cả linear projection không có bias; riêng convolution của Mamba-2 mặc định `conv_bias: true`, được ghi rõ trong config. Mamba-2 dùng một nhóm B/C (`ngroups=1`), `d_ssm = expand × hidden_size`, skip D theo head và RMSNorm với gate trước normalization theo cấu hình upstream tương ứng.

| Config | Tham số thực tế | Vai trò |
|---|---:|---|
| `debug.yaml` | 209.048 | Kiểm thử CPU, byte tokenizer 264 token |
| `hybrid_59m.yaml` | 59.291.760 | Pilot trước khi chạy model lớn |
| `hybrid_373m.yaml` | 373.381.824 | Candidate theo các kích thước trong tài liệu đính kèm |
| `transformer_373m.yaml` | 372.559.872 | Pure GQA control; ít hơn khoảng 0,22% tham số |
| `hybrid_355m.yaml` | 354.507.456 | Variant giảm FFN từ 2816 xuống 2560 |
| `transformer_355m.yaml` | 353.685.504 | Control cho variant 355M; ít hơn khoảng 0,23% |

**Cấu hình gọi là “350M” trong bản thiết kế thực ra gần 373M** với FFN 2816 và các lựa chọn triển khai trên. Repo không lặng lẽ đổi cấu hình để khớp tên. Đếm lại mà không cấp phát trọng số lớn:

```bash
python -m scripts.model_summary configs/hybrid_373m.yaml configs/transformer_373m.yaml
```

Hai backend được chọn tường minh:

- `mamba2`: import `Mamba2` từ `mamba_ssm`, dùng implementation CUDA/Triton upstream. Thiếu dependency thì báo lỗi, không tự thay bằng RNN khác.
- `reference`: bản PyTorch tuần tự của tập cấu hình Mamba-2 nêu trên; hỗ trợ gradient và cache để kiểm thử. Không phải kernel SSD song song, không dùng để so sánh tốc độ sản xuất. Có bài kiểm tra parity với upstream, nhưng cần GPU để chạy.

Layer cuối của hybrid là attention. Pure Transformer có cùng hidden size, số layer, head và tokenizer; tăng chiều FFN để ngân sách tham số gần nhau. Đây là đối chứng gần khớp tham số, **không phải đối chứng có FLOPs bằng nhau**.

## 2. Cài đặt và chạy thử offline

Dùng Python 3.10 trở lên; nên chọn môi trường Python/CUDA có wheel phù hợp trên máy GPU. Môi trường thực sự đã kiểm thử của gói này: Python 3.13.5, PyTorch 2.10.0+cpu. Xem `reports/VALIDATION.md`.

```bash
unzip scholartiny_hybrid_minimind.zip
cd scholartiny
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Chỉ chạy smoke test/CPU:
python -m pip install -e '.[test]'
python -m pytest -q
python -m scripts.smoke_test --out .work/my_smoke
```

Smoke test tự tạo dữ liệu giả nhỏ, byte tokenizer, chạy 2 bước pretrain, 2 bước SFT, tạo chỉ mục RAG, thực thi phép tính trong tiến trình riêng và sinh vài token. **Nội dung sinh ra chưa có ý nghĩa**; đây chỉ là kiểm thử đường đi của chương trình.

Không có fallback byte tokenizer âm thầm khi thiếu BPE. `--debug-byte` phải được chọn tường minh. Sau khi đã cài dependencies, smoke test không cần mạng. Dùng một thư mục `--out` mới cho mỗi lần chạy để không trộn artifacts cũ.

### Máy CUDA dùng Mamba-2 thật

Cài PyTorch CUDA tương thích GPU/driver của máy trước. Không dùng bản CPU bên trên cho huấn luyện GPU. Sau đó:

```bash
python -m pip install -e '.[data,test]'
python -m pip install packaging ninja einops
python -m pip install 'causal-conv1d>=1.4,<2' --no-build-isolation
python -m pip install 'mamba-ssm>=2.2,<3' --no-build-isolation

python -c 'import torch; from mamba_ssm import Mamba2; print(torch.__version__, torch.cuda.is_available())'
python -m pytest -q -m gpu
python -m pip freeze > reports/environment-local.txt
```

Khoảng phiên bản trên là interface mục tiêu, không phải cam kết mọi tổ hợp CUDA/driver đều hoạt động. Không chạy pretrain lớn trước khi hai GPU test qua. Không đưa ra kết quả GPU benchmark trong gói này vì môi trường tạo gói không có CUDA.

## 3. Thu thập public dataset

`configs/data_sources.yaml` chứa registry nguồn và adapter, không chứa dữ liệu tải sẵn.

| Tên nguồn trong repo | Dataset / subset | Giai đoạn |
|---|---|---|
| `fineweb_edu` | `HuggingFaceTB/smollm-corpus`, `fineweb-edu-dedup` | Pretrain văn bản giáo dục |
| `cosmopedia` | `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2`; lọc các format textbook đã khai báo | Pretrain nội dung giáo dục tổng hợp |
| `wikipedia_vi` | `wikimedia/wikipedia`, `20231101.vi` | Pretrain tiếng Việt, nguồn RAG |
| `openmath` | `nvidia/OpenMathInstruct-2`, split `train_1M` | SFT toán |
| `xlam` | `Salesforce/xlam-function-calling-60k` | SFT chọn tool và arguments |
| `ultrachat` | `HuggingFaceH4/ultrachat_200k`, `train_sft` | SFT chỉ dẫn và hội thoại |

Dataset card được đối chiếu ngày 10-09-2026. Xem liên kết nguồn và lưu ý trong `docs/SOURCES.md`, `docs/DATA_POLICY.md`. **Public không đồng nghĩa được tái sử dụng vô điều kiện.** Collector yêu cầu người chạy đọc điều khoản và xác nhận `--acknowledge-source-terms`. Cờ này chỉ ghi nhận quyết định của người chạy, không kiểm chứng quyền sử dụng.

xLAM yêu cầu chấp nhận điều khoản truy cập trên Hugging Face. Token đặt bằng biến môi trường `HF_TOKEN`; không ghi vào config/git. Phải xem cả văn bản điều khoản, không chỉ nhãn license: card xLAM có thêm mô tả research-purpose. Wikipedia cần lưu attribution/điều kiện áp dụng. SmolLM-Corpus có license ở mức dataset; không suy diễn rằng mọi webpage/code được nhúng đều có cùng quyền.

```bash
# Dùng HF_TOKEN từ môi trường khi nguồn yêu cầu đăng nhập.
# Sau khi đã đọc/chấp thuận điều khoản nguồn:
python -m dataset.collect \
  --sources fineweb_edu cosmopedia wikipedia_vi \
  --limit 10000 \
  --acknowledge-source-terms

python -m dataset.collect \
  --sources openmath xlam ultrachat \
  --limit 10000 \
  --acknowledge-source-terms
```

`--limit` là số mẫu đã chấp nhận tối đa **trên mỗi nguồn**, bao gồm các mẫu đã có khi resume; không phải số token và không phải tải toàn bộ dataset. Có giới hạn số hàng quét, bộ đệm shuffle hữu hạn và bộ lọc chiều dài. Shuffle buffer không tạo mẫu ngẫu nhiên đều từ toàn bộ corpus hàng trăm tỷ token.

Tăng quy mô sau lần thử:

```bash
python -m dataset.collect \
  --sources fineweb_edu cosmopedia wikipedia_vi \
  --limit 100000 --resume \
  --acknowledge-source-terms
```

Collector lưu `data/collection.sqlite` và xuất:

```text
data/normalized/<source>/
  train.jsonl
  val.jsonl
  test.jsonl
  manifest.json
```

Manifest giữ dataset ID, subset, upstream split, commit SHA được resolve, dataset card, license card và thống kê giữ/loại. Từng record giữ provenance gồm URL/ID/title khi nguồn cung cấp. DB giữ ledger trùng lặp giữa nguồn. `resume` dùng đúng snapshot và cấu hình cũ; đổi filter, seed hoặc deny list phải dùng DB/output mới.

Pipeline sử dụng **exact dedup** toàn cục theo nội dung và chia split theo `group_id` trước tokenization. Mọi lời giải OpenMath của cùng bài toán nằm trong một split. Có chặn overlap chuẩn hóa và 13-word n-gram với đề benchmark do người chạy cung cấp:

```bash
# Tạo bộ dữ liệu mới khi thay đổi deny list, tránh giữ lại dữ liệu cũ đã nhiễm.
python -m dataset.collect \
  --sources openmath --limit 10000 \
  --db data/clean_collection.sqlite --out data/clean_normalized \
  --deny-file benchmarks/heldout_questions.jsonl \
  --acknowledge-source-terms
```

Mỗi dòng deny file có `problem`, `question` hoặc `text`. Không đưa các benchmark test vào training. Exact/n-gram filter **không bảo đảm** sạch paraphrase hoặc trùng ngữ nghĩa. Repo chưa triển khai MinHash toàn corpus, bộ lọc PII hoàn chỉnh hay audit pháp lý.

## 4. Tạo dữ liệu academic agent và RAG

xLAM cung cấp target cho function-call plan; không tự tạo đầy đủ quan sát và câu trả lời cuối cho các tool nội bộ của repo. Bổ sung các hội thoại có kết quả thực:

```bash
python -m scripts.generate_academic \
  --count 20000 --seed 123 --out data/normalized

python -m scripts.build_rag_sft \
  --input-root data/normalized/wikipedia_vi \
  --out data/normalized/rag_grounded --limit 10000
```

Generator academic tạo bài tính số chính xác, đạo hàm, trị riêng, chuỗi hai lượt gọi tool, no-tool, RAG có distractor và thiếu bằng chứng. Kết quả toán của các ví dụ được tính từ hàm allowlist, không do LLM bịa. Các ví dụ này là bài tập đơn giản; không phải dữ liệu suy luận nghiên cứu sâu hoặc tập chứng minh hình thức.

Generator RAG từ public documents chỉ tạo **bài chọn đoạn/trích nguyên văn/citation/thiếu bằng chứng**, không giả vờ tự hiểu paper rồi tạo semantic QA đúng. Nó kế thừa split và provenance của tài liệu gốc, chỉ lấy distractor trong cùng split. Giữ document group chung giữa pretrain và RAG warmup.

Schema hội thoại chuẩn:

```json
{
  "kind": "sft",
  "tools": [{"name": "math.calculate", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}],
  "messages": [
    {"role": "user", "content": "Tính 1/3 + 1/6."},
    {"role": "assistant", "content": "", "tool_calls": [{"name": "math.calculate", "arguments": {"expression": "1/3+1/6"}}]},
    {"role": "tool", "name": "math.calculate", "content": "{\"ok\":true,\"result\":{\"exact\":\"1/2\"}}"},
    {"role": "assistant", "content": "Kết quả chính xác là 1/2."}
  ]
}
```

Cùng serializer được dùng ở SFT và inference. Tool call được sinh bằng `<tool_call>{"name":...,"arguments":...}</tool_call>`. Role headers và tool observations không phải target loss; assistant content, arguments và token kết thúc lượt mới được học. Các delimiter dành riêng xuất hiện trong external text được escape để không trở thành role-control token.

## 5. Huấn luyện tokenizer 32K

```bash
python -m scripts.train_tokenizer \
  --input \
    data/normalized/fineweb_edu/train.jsonl \
    data/normalized/cosmopedia/train.jsonl \
    data/normalized/wikipedia_vi/train.jsonl \
  --vocab-size 32768 --out model/tokenizer.json
```

Đây là byte-level BPE có đủ bảng chữ cái byte để biểu diễn Unicode, tiếng Việt và công thức; không đồng nghĩa token hóa mọi công thức đều tối ưu. Tokenizer chỉ học trên `train.jsonl`, không lấy val/test. CLI in `actual_vocab_size`; corpus quá nhỏ có thể chưa tạo đủ 32768 token. Config phải dùng đúng kích thước thực tế, và H/T phải dùng cùng tokenizer. Training runner kiểm tra cả kích thước lẫn fingerprint, không tự resize embedding.

Chưa có tokenizer BPE được huấn luyện sẵn trong archive. Byte tokenizer ở smoke test là bộ kiểm thử, không dùng cho mô hình academic 59M/373M.

## 6. Tokenize dữ liệu

```bash
for source in fineweb_edu cosmopedia wikipedia_vi; do
  for split in train val test; do
    python -m dataset.prepare \
      --input data/normalized/$source/$split.jsonl \
      --tokenizer model/tokenizer.json \
      --out data/packed/$source/$split \
      --kind pretrain --max-seq-len 8192
  done
done

for source in openmath xlam ultrachat academic_synthetic rag_grounded; do
  for split in train val test; do
    python -m dataset.prepare \
      --input data/normalized/$source/$split.jsonl \
      --tokenizer model/tokenizer.json \
      --out data/packed/$source/$split \
      --kind sft --max-seq-len 4096
  done
done
```

Thư mục output phải mới/rỗng. Khi thay đổi tokenizer hoặc giới hạn SFT, tạo thư mục tokenized mới thay vì trộn artifacts. Split quá nhỏ hoặc toàn mẫu vượt giới hạn sẽ báo lỗi; tăng lượng mẫu thu thập hoặc xem `manifest.json`, không biến split rỗng thành kết quả tốt.

Pretrain là chuỗi văn bản nối bằng EOS và lấy block cố định. **EOS không reset SSM và không cách ly attention giữa tài liệu trong cùng block**; đây là chính sách continuous language modeling có chủ đích. State reset giữa các batch forward. SFT không nối các hội thoại khác nhau: một hội thoại mỗi hàng, right padding; sample vượt giới hạn bị loại có thống kê, không cắt mất tool schema/observation rồi vẫn học câu trả lời.

## 7. Pretrain rồi full SFT

Khởi đầu pilot 59M để kiểm tra gradient, loss, chất lượng mẫu và memory trước:

```bash
torchrun --standalone --nproc_per_node=1 -m trainer.train_pretrain \
  --config configs/hybrid_59m.yaml --tokenizer model/tokenizer.json \
  --data configs/mix_pretrain.yaml \
  --val-data data/packed/fineweb_edu/val \
  --out out/hybrid59_pretrain \
  --device cuda --precision bf16 \
  --seq-len 2048 --batch-size 2 --grad-accum 16 \
  --max-steps 10000 --warmup-steps 500 --lr 3e-4 \
  --gradient-checkpointing --save-every 200 --eval-every 200
```

Bản chính đổi thành `configs/hybrid_373m.yaml`; đối chứng đổi thành `configs/transformer_373m.yaml`. Chạy hai nhánh với cùng dữ liệu đã đóng băng, tokenizer, seed sampling, lịch context và ngân sách token. Config `hybrid_355m.yaml` là một thí nghiệm khác, không đổi architecture giữa chừng để tiếp tục cùng checkpoint.

```bash
torchrun --standalone --nproc_per_node=1 -m trainer.train_full_sft \
  --config configs/hybrid_59m.yaml --tokenizer model/tokenizer.json \
  --data configs/mix_sft.yaml \
  --val-data data/packed/academic_synthetic/val \
  --init out/hybrid59_pretrain/last.pt \
  --out out/hybrid59_sft \
  --device cuda --precision bf16 \
  --seq-len 4096 --batch-size 1 --grad-accum 32 \
  --max-steps 3000 --warmup-steps 150 --lr 5e-5 \
  --gradient-checkpointing --save-every 100 --eval-every 100
```

Các số bước/LR/tỷ lệ nguồn là **điểm khởi đầu thí nghiệm**, không phải recipe đã tối ưu. Với data budget nhỏ, replacement sampling sẽ lặp lại mẫu nhiều lần; không nên gọi số token nhìn thấy là số token độc nhất. Tỷ lệ `mix_pretrain.yaml` gần tỷ lệ token vì block dài như nhau; tỷ lệ `mix_sft.yaml` là tỷ lệ **example**, không phải tỷ lệ token.

Runner có AdamW, cosine LR, warmup, BF16 autocast, gradient accumulation/checkpointing, global-token-normalized SFT loss dưới DDP, gradient clipping và loss projection chia chunk. Bỏ weight decay cho A_log, D, dt_bias; không ghi đè initialization riêng của Mamba. Master parameters/optimizer ở FP32. Không dùng FP16/GradScaler trong phiên bản này.

Checkpoint lưu model, optimizer, step, số token, LR schedule, tokenizer fingerprint và RNG/sampler từng rank. `--resume` tiếp tục chính xác với cùng schedule, data/shape và world size. `--init` chỉ lấy trọng số để mở một stage mới. Không hứa exact resume khi đổi số GPU. Ví dụ dùng lại toàn bộ tham số của lệnh pretrain và thêm:

```bash
--resume out/hybrid59_pretrain/last.pt
```

Log lưu vào `metrics.jsonl`; `val_ppl` trên SFT chỉ tính assistant targets, không so ngang với PPL toàn token của pretrain. `--token-budget` có thể dừng sau một ngưỡng supervised token, với sai lệch tối đa một global batch; LR vẫn được xác định bởi `--max-steps` nên cần đặt hai ngân sách nhất quán.

## 8. Agent và RAG local

```bash
python -m scripts.build_rag_index \
  --input data/normalized/wikipedia_vi/train.jsonl \
  --out data/rag.sqlite --limit 10000

python -m scripts.chat \
  --checkpoint out/hybrid59_sft/last.pt \
  --tokenizer model/tokenizer.json \
  --device cuda --precision bf16 --agent --rag-db data/rag.sqlite \
  --prompt 'Tính chính xác 1/3 + 1/6 bằng công cụ.'
```

Tool runtime chỉ thực thi `math.calculate`, `math.differentiate`, `math.eigenvalues` và `retrieve`. Các function name ngoài allowlist, bao gồm các API khác model có thể đã thấy trong xLAM, bị từ chối. Không chạy arbitrary Python, `eval`, `exec`, `sympify` từ string, shell hay network do model chỉ định. Parser toán dựng cây qua AST allowlist và có giới hạn độ sâu, số nút, số mũ, kích thước ma trận; phép tính chạy trong tiến trình riêng có timeout.

Đây không phải sandbox multi-tenant đã audit: sản xuất cần thêm container/OS isolation và quota bộ nhớ. Prompt instruction rằng tool output là untrusted data không tự bảo đảm chống mọi prompt injection.

RAG dùng SQLite FTS5/BM25 để giữ dependency gọn. Chưa có embedding, reranker, vector DB hay GraphRAG. Chỉ mục giữ chunk ID/source/URL. Citation checker kiểm tra ID có được quan sát hay không, **không chứng minh entailment hoặc tính đúng của phát biểu**. Mỗi lượt tool mới được prefill lại; trong một lượt sinh token có KV + SSM cache. Generation hiện hỗ trợ batch 1 không padding.

## 9. Đánh giá H so với T

```bash
python -m scripts.evaluate \
  --checkpoint out/hybrid59_sft/last.pt --tokenizer model/tokenizer.json \
  --input data/normalized/academic_synthetic/test.jsonl \
  --out out/eval_h59 --device cuda --precision bf16 --limit 100

python -m scripts.benchmark \
  --config configs/hybrid_373m.yaml --device cuda --precision bf16 \
  --lengths 2048 8000 --decode-tokens 32
```

`evaluate` báo tool-plan parseability, tool selection, name+arguments exact match, answer string exact match và citation ID precision/recall. Các bài completion sử dụng **gold history/gold tool observations**; không được gọi đây là end-to-end agent success, semantic faithfulness hay MATH accuracy. Agent CLI có trace thực thi thật, nhưng benchmark vòng kín đầy đủ cho mọi nguồn chưa được đóng gói.

`benchmark` đo kernel runtime của random weights, không đánh giá chất lượng. Tách prefill/decode; loại warmup; báo peak allocated CUDA memory. Dùng KV cache `torch.cat` dễ đọc, chưa phải paged/preallocated serving tối ưu. Mặc định 8000 token để chừa chỗ decode trong context 8192. Mốc 16K cần cấu hình/curriculum tương ứng; không tự coi model train 2K là hiểu tốt 16K.

Ở batch 1, cache 2-byte, config 373M và 8K: phần KV lý thuyết là 48 MiB cho hybrid so với 192 MiB cho pure GQA; hybrid thêm khoảng 4,8 MiB state/convolution cache. **Không phải tổng VRAM giảm 4 lần**. Hybrid vẫn có KV tăng theo context và global attention prefill có thành phần bậc hai. Có thể xem công thức và số đếm trong `reports/model_parameters.json`.

Trước quyết định architecture, cần thêm đánh giá độc lập về associative/exact recall, mathematical correctness, citation entailment, retrieval robustness, tool execution success và ít nhất vài seed. Mamba-2/H là candidate, không được chứng minh là tối ưu cho bài toán này.

## 10. Phạm vi bàn giao

Đã có code thực thi cho model, collector, chuẩn hóa/group split/dedup, tokenizer, pretrain/SFT, checkpoint, công cụ, lexical RAG và diagnostics. Có test CPU/optional GPU, ví dụ gốc nhỏ và báo cáo chạy thực tế.

Chưa có trọng số academic đã train, public corpus tải kèm, tokenizer BPE đã train, kết quả CUDA thực đo, LoRA/DPO/GRPO, speculative decoding, constrained JSON generation, export GGUF, Hugging Face AutoModel integration hoặc tương thích sẵn vLLM/Ollama. Không nên đổi `model_type` để giả nhận một kiến trúc khác trong các serving engine.

Xem `docs/ARCHITECTURE.md`, `docs/DATA_POLICY.md`, `docs/EXPERIMENT_PLAN.md`, `docs/SOURCES.md` và `reports/VALIDATION.md` để triển khai và audit.
