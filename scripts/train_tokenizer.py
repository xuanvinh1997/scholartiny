import argparse
from dataset.tokenizer import train_bpe,ByteTokenizer

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',nargs='+');p.add_argument('--out',required=True)
    p.add_argument('--vocab-size',type=int,default=32768);p.add_argument('--debug-byte',action='store_true')
    a=p.parse_args()
    if a.debug_byte:ByteTokenizer().save(a.out);print('Debug UTF-8 byte tokenizer; vocabulary=264')
    else:
        if not a.input:p.error('--input is required for BPE')
        print('Actual vocabulary:',train_bpe(a.input,a.out,a.vocab_size))
if __name__=='__main__':main()
