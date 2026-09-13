import argparse,json
from agent.rag import build_index

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    print(json.dumps(build_index(a.input,a.out,limit=a.limit)))
if __name__=='__main__':main()
