**Transformer + SSM hybrid** nghĩa là trong cùng một LLM, ta xen kẽ hai cơ chế xử lý chuỗi:

```math
\boxed{\text{Attention} + \text{State Space Model}}
```

Thay vì 24 layer đều là Transformer attention.

Ví dụ ScholarTiny có thể là:

```text
Embedding
   │
   ▼
┌────────────────────────────┐
│ Transformer Attention      │
│ SwiGLU FFN                 │
├────────────────────────────┤
│ Mamba-2 / SSM              │
│ SwiGLU FFN                 │
├────────────────────────────┤
│ Transformer Attention      │
│ SwiGLU FFN                 │
├────────────────────────────┤
│ Mamba-2 / SSM              │
│ SwiGLU FFN                 │
└────────────────────────────┘
              × N
```

Hai loại block làm hai việc khá khác nhau.

**Transformer attention** mạnh ở việc truy cập chính xác một token hoặc đoạn cụ thể trong context:

```math
y_i=\sum_j \operatorname{softmax} \left( \frac{q_i k_j^T}{\sqrt d} \right)v_j
```

Ví dụ câu cuối paper hỏi:

> “Using Eq. (17), calculate the logical error rate.”

Attention có thể trực tiếp liên kết:

```text
"Eq. (17)"
      │
      └──────────────► phương trình ở rất xa phía trước
```

Đây là dạng **content-addressable memory** rất hữu ích cho RAG, citation, tool results và toán học.

Trong khi đó, SSM/Mamba xử lý chuỗi theo trạng thái:

```math
h_t=A_t h_{t-1}+B_t x_t
```

```math
y_t=C_t h_t
```

Có thể hình dung:

```text
x1 → h1 → h2 → h3 → ... → ht
      ↑     ↑             ↑
      x2    x3            xt
```

Thay vì giữ toàn bộ KV của mọi token, nó duy trì một **compressed state**.

Đó là lý do Mamba rất hấp dẫn cho long-context: inference có thể có state cố định thay vì KV cache tăng tuyến tính theo sequence. Mamba-2 còn đưa ra State Space Duality nối quan hệ giữa attention và SSM và báo cáo core layer nhanh hơn Mamba ban đầu trong các thiết lập của paper. ([arXiv](https://arxiv.org/abs/2405.21060?utm_source=chatgpt.com "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"))

### Vì sao không dùng toàn SSM?

Đây là điểm quan trọng.

Suppose RAG trả về:

```text
[Document A]
...
Eq. 4:
E_J = Φ₀ I_c / 2π
...

[Document B]
...

[Document C]
...
```

Sau 8,000 token model được hỏi:

```text
What was the relation between EJ and Ic in document A?
```

Attention có một cơ chế tự nhiên:

```math
q_{\text{question}} \rightarrow k_{\text{Eq.4}}
```

rồi retrieve trực tiếp value tương ứng.

SSM phải giữ thông tin đó trong:

```math
h_t
```

suốt quá trình đọc hàng nghìn token.

Do đó có nguy cơ:

```math
\text{exact recall} < \text{attention}
```

Các nghiên cứu về efficient sequence models cũng chỉ ra rằng recall/associative retrieval là một trong những bài toán phân biệt quan trọng giữa attention và các recurrent/convolutional alternatives. ([arXiv](https://arxiv.org/pdf/2312.04927?utm_source=chatgpt.com "Zoology: Measuring and Improving Recall in Efficient Language"))

Với **academic RAG**, exact recall cực kỳ quan trọng.

Cho nên:

```math
\boxed{\text{Pure Mamba không phải lựa chọn của tôi}}
```

---

## Còn pure Transformer?

Pure Transformer có:

```math
O(n^2)
```

attention compute nếu full attention.

Và autoregressive inference phải giữ KV:

```math
KV\ cache \propto n.
```

Khi context chứa:

```text
system prompt
+ question
+ 5 chunks RAG
+ previous reasoning
+ Python output
+ SymPy output
+ citations
```

n có thể nhanh chóng lên 8K–32K.

Tiny model lúc này nhỏ về weights nhưng **không còn nhỏ về runtime memory**.

Hybrid giải quyết chính xác bài toán này.

Jamba là ví dụ thực tế nổi bật: họ xen kẽ Transformer và Mamba để kết hợp khả năng của attention với memory/throughput của SSM, và báo cáo lợi thế long-context đáng kể ở scale lớn. ([arXiv](https://arxiv.org/abs/2403.19887?utm_source=chatgpt.com "Jamba: A Hybrid Transformer-Mamba Language Model"))

---

# Với ScholarTiny, tôi sẽ không xen kẽ 1:1

Ví dụ này:

```text
Attention
Mamba
Attention
Mamba
Attention
Mamba
```

dễ hiểu nhưng chưa chắc tối ưu.

Tôi sẽ thử:

```math
\boxed{3\ \mathrm{SSM}:1\ \mathrm{Attention}}
```

tức:

```text
Mamba-2
Mamba-2
Mamba-2
Global GQA

Mamba-2
Mamba-2
Mamba-2
Global GQA

...
```

Với 24 layers:

```text
18 × Mamba/SSM
6 × Attention
```

Lý do: SSM làm phần lớn sequential processing rẻ; attention xuất hiện định kỳ để cung cấp **explicit token-to-token retrieval**.

Đây gần với triết lý Jamba, mặc dù tỷ lệ chính xác cần benchmark ở scale 350M chứ không nên copy từ model hàng chục tỷ parameters. ([arXiv](https://arxiv.org/abs/2403.19887?utm_source=chatgpt.com "Jamba: A Hybrid Transformer-Mamba Language Model"))

---

## Nhưng tôi sẽ sửa thêm: attention nên là global

Ở kiến trúc hybrid, tôi không còn quá thích:

```text
Local Attention
+
SSM
```

vì SSM vốn đã đảm nhiệm rất nhiều local/sequential propagation.

Tôi muốn dùng số layer attention ít hơn nhưng để chúng làm:

```math
\boxed{\text{Global Attention}}
```

Ví dụ:

```text
tokens 1................................8192
           ↑
           │
       attention
           │
           ↓
token 8192 can inspect any previous token
```

Như vậy phân công rất sạch:

| ComponentVai trò |                           |
| ---------------- | ------------------------- |
| SSM              | sequence processing       |
| SSM              | compressed long memory    |
| SSM              | cheap context propagation |
| Global Attention | precise retrieval         |
| Global Attention | equations/citations       |
| Global Attention | RAG evidence linking      |

---

# Một block ScholarTiny có thể như sau

SSM block:

```math
x'=x+\operatorname{Mamba2}(\operatorname{RMSNorm}(x))
```

```math
y=x'+\operatorname{SwiGLU}(\operatorname{RMSNorm}(x'))
```

Attention block:

```math
x'=x+\operatorname{GQA} ( \operatorname{RMSNorm}(x) )
```

```math
y=x'+\operatorname{SwiGLU}(\operatorname{RMSNorm}(x'))
```

Và Attention dùng:

```math
Q=\operatorname{RMSNorm}(XW_Q)
```

```math
K=\operatorname{RMSNorm}(XW_K)
```

rồi RoPE.

---

# Tôi sẽ ưu tiên Mamba-2 hơn Mamba-1

Nếu viết architecture mới năm 2026, tôi không có nhiều lý do để bắt đầu bằng Mamba-1.

Mamba-2 xuất phát từ **State Space Duality (SSD)**, đưa attention và structured state-space models vào một framework gần nhau hơn, đồng thời paper báo cáo Mamba-2 core layer nhanh hơn Mamba ban đầu khoảng 2–8× trong các benchmark của họ. ([arXiv](https://arxiv.org/abs/2405.21060?utm_source=chatgpt.com "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"))

Vậy architecture sẽ là:

```math
\boxed{ \text{GQA Transformer} + \text{Mamba-2} }
```

chứ không phải generic SSM.

---

# Một candidate 350M thực tế

Tôi sẽ bắt đầu gần như sau:

```yaml
model_type: scholar_hybrid

vocab_size: 32768

hidden_size: 1024
num_hidden_layers: 24

# Transformer
num_attention_layers: 6
num_attention_heads: 16
num_key_value_heads: 4
head_dim: 64
qk_norm: true
rope: true

# SSM
num_ssm_layers: 18
ssm_type: mamba2
state_size: 64
conv_kernel: 4
expand: 2

# FFN
hidden_act: swiglu
intermediate_size: 2816

# common
norm: rmsnorm
tie_word_embeddings: true
bias: false

context_length: 8192
```

Layer pattern:

```text
00 Mamba
01 Mamba
02 Mamba
03 Attention

04 Mamba
05 Mamba
06 Mamba
07 Attention

08 Mamba
09 Mamba
10 Mamba
11 Attention

12 Mamba
13 Mamba
14 Mamba
15 Attention

16 Mamba
17 Mamba
18 Mamba
19 Attention

20 Mamba
21 Mamba
22 Mamba
23 Attention
```

Hay ngắn gọn:

```math
\boxed{ (M,M,M,A)\times6 }
```

Đây là architecture tôi thấy rất đáng benchmark.

---

# Một chi tiết còn quan trọng hơn: layer cuối

Tôi muốn layer cuối là **Attention**, không phải SSM:

```text
...
Mamba
Mamba
Mamba
Attention
RMSNorm
LM Head
```

Vì ngay trước khi sinh token:

```text
<tool_call>
```

hoặc:

```text
According to Eq. (14)...
```

model có cơ hội cuối cùng truy cập trực tiếp toàn context.

Đặc biệt hữu ích cho:

```text
tool arguments
citation IDs
numerical results
variable names
retrieved evidence
```

---

# Với tool calling, hybrid khá thú vị

Ví dụ context:

```text
User:
Compute eigenvalues of matrix A.

          ↓

SSM layers:
understand sequence/intention

          ↓

Attention:
focus on matrix values

          ↓

SSM:
process reasoning state

          ↓

Attention:
identify appropriate tool

          ↓

<tool_call>
{"name":"python",...}

          ↓

Tool result

          ↓

SSM:
integrate observation

          ↓

Final Attention:
re-read exact tool result

          ↓

Answer
```

Đây chính xác là dạng workload mà hybrid architecture có lý về mặt thiết kế.

---

# RAG còn phù hợp hơn

Một query có thể trở thành:

```text
Question
   │
   ▼
<tool_call>retrieve(...)</tool_call>
   │
   ▼
Chunk A
Chunk B
Chunk C
Chunk D
   │
   ▼
Mamba state
   │
   ▼
Global Attention
   │
   ├── Chunk A: relevant
   ├── Chunk C: relevant
   └── Chunk B/D: ignore
   │
   ▼
reason
```

Ta đang kết hợp:

```math
\underbrace{\text{compressed sequence memory}}_{\mathrm{SSM}} + \underbrace{\text{exact retrieval}}_{\mathrm{Attention}}
```

Đó là lý do tôi cho hybrid điểm cao đối với mục tiêu của bạn.

---

## Nhưng tôi sẽ làm **hai model song song**

Đây là phần tôi cho là bắt buộc nếu dự án nghiêm túc:

```text
ScholarTiny-T
350M pure Transformer

ScholarTiny-H
350M Transformer + Mamba-2
```

Giữ nguyên:

```text
tokenizer
dataset
parameter count
training tokens
optimizer
LR schedule
SFT
```

rồi đo:

```math
\begin{aligned} &\text{Validation PPL}\\ &\text{MATH accuracy}\\ &\text{exact recall}\\ &\text{RAG faithfulness}\\ &\text{citation accuracy}\\ &\text{tool selection}\\ &\text{tool argument accuracy}\\ &\text{tokens/sec}\\ &\text{VRAM @ 2K,8K,16K} \end{aligned}
```

Nếu Hybrid chỉ nhanh hơn nhưng math/RAG recall giảm rõ rệt:

```math
\rightarrow \text{bỏ SSM}.
```

Nếu chất lượng gần Transformer nhưng:

```math
VRAM\downarrow,\qquad throughput\uparrow
```

thì Hybrid trở thành architecture chính.

Do bằng chứng hiện có về Jamba/Mamba chủ yếu ở các scale khác, **không nên coi** **`(M,M,M,A)×6`** **là tối ưu đã được chứng minh cho 350M**. Nó là candidate mạnh cần ablation. ([arXiv](https://arxiv.org/abs/2405.21060?utm_source=chatgpt.com "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"))

Với dự án **tiny academic agent** của bạn, tôi hiện sẽ chọn **`Mamba-2 ×3 → Global GQA ×1`** **làm kiến trúc experimental chính**, và một **Qwen3-like pure GQA Transformer cùng parameter budget làm control baseline**. Đây là cách đủ mới về kiến trúc nhưng vẫn giữ được tính khoa học khi đánh giá.