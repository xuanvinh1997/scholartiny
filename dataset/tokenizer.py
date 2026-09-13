"""Trainable byte-level BPE; dependency-free byte tokenizer for offline smoke tests."""
import hashlib
import json
from pathlib import Path

SPECIALS = ['<|pad|>','<|unk|>','<|bos|>','<|eos|>','<|im_start|>','<|im_end|>',
            '<tool_call>','</tool_call>']

def literal(text):
    # Reserved delimiters in external/user text must not become role-control tokens.
    text = str(text)
    for special in SPECIALS:
        text = text.replace(special,special.replace('<','＜'))
    return text

class ByteTokenizer:
    """264-token UTF-8 codec. Tests only, not the proposed 32K production tokenizer."""
    vocab_size = 256+len(SPECIALS)
    def token_id(self,token): return SPECIALS.index(token)
    def encode(self,text): return [v+len(SPECIALS) for v in literal(text).encode('utf-8')]
    def decode(self,ids):
        result,buf = [],bytearray()
        for i in ids:
            if i < len(SPECIALS):
                result.append(buf.decode('utf-8',errors='replace')); buf.clear()
                result.append(SPECIALS[i])
            else: buf.append(i-len(SPECIALS))
        result.append(buf.decode('utf-8',errors='replace'))
        return ''.join(result)
    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps({'type':'scholartiny_byte_debug','version':1}),encoding='utf-8')

class BPETokenizer:
    def __init__(self,path):
        from tokenizers import Tokenizer
        self.core = Tokenizer.from_file(str(path))
        self.vocab_size = self.core.get_vocab_size()
        if any(self.core.token_to_id(s) is None for s in SPECIALS):
            raise ValueError('Tokenizer lacks required protocol tokens')
    def token_id(self,token): return self.core.token_to_id(token)
    def encode(self,text): return self.core.encode(literal(text),add_special_tokens=False).ids
    def decode(self,ids): return self.core.decode(list(ids),skip_special_tokens=False)

def load_tokenizer(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    return ByteTokenizer() if obj.get('type')=='scholartiny_byte_debug' else BPETokenizer(path)

def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def train_bpe(paths,out,vocab_size=32768,min_frequency=2):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    if vocab_size < len(SPECIALS)+256:
        raise ValueError('vocab_size must accommodate all 256 bytes and protocol tokens')
    core = Tokenizer(models.BPE(unk_token='<|unk|>'))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    def texts():
        for p in paths:
            if Path(p).name != 'train.jsonl':
                raise ValueError('Tokenizer training accepts only train.jsonl, never val/test')
            with open(p,encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line)
                    if row.get('split') not in (None,'train'):
                        raise ValueError('Non-training row in tokenizer corpus')
                    if row.get('text'): yield literal(row['text'])
                    for message in row.get('messages',[]):
                        yield literal(message.get('content',''))
                    if row.get('tools'): yield json.dumps(row['tools'],ensure_ascii=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size,min_frequency=min_frequency,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                 special_tokens=SPECIALS,show_progress=True)
    core.train_from_iterator(texts(),trainer=trainer)
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    core.save(str(out))
    Path(str(out)+'.meta.json').write_text(json.dumps({
        'requested_vocab_size':vocab_size,'actual_vocab_size':core.get_vocab_size(),
        'inputs':[str(p) for p in paths],'sha256':fingerprint(out)},indent=2),encoding='utf-8')
    return core.get_vocab_size()
