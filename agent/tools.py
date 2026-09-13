"""Allowlisted academic tools. No eval/exec/sympify, shell, arbitrary files or network.

The process timeout is defense in depth, not a multi-tenant security boundary.
Production deployments should add a separate container/OS sandbox and quotas.
"""
import ast
import math
import multiprocessing as mp
import sys
from typing import Any
import jsonschema

SCHEMAS = [
 {'name':'math.calculate','description':'Evaluate a bounded mathematical expression.',
  'parameters':{'type':'object','properties':{'expression':{'type':'string','maxLength':256}},
                'required':['expression'],'additionalProperties':False}},
 {'name':'math.differentiate','description':'Differentiate a bounded expression with respect to x.',
  'parameters':{'type':'object','properties':{'expression':{'type':'string','maxLength':256}},
                'required':['expression'],'additionalProperties':False}},
 {'name':'math.eigenvalues','description':'Compute eigenvalues of a square numeric matrix up to 4 by 4.',
  'parameters':{'type':'object','properties':{'matrix':{'type':'array','minItems':1,'maxItems':4,
       'items':{'type':'array','minItems':1,'maxItems':4,
                'items':{'type':'number','minimum':-1000000,'maximum':1000000}}}},
                'required':['matrix'],'additionalProperties':False}},
 {'name':'retrieve','description':'Search the local academic corpus. Results include chunk IDs for citations.',
  'parameters':{'type':'object','properties':{'query':{'type':'string','minLength':1,'maxLength':256},
                            'top_k':{'type':'integer','minimum':1,'maximum':5}},
                'required':['query'],'additionalProperties':False}}
]

def safe_expression(text,allow_x=False):
    """Build a SymPy tree through an explicit Python AST allowlist; never evaluate code."""
    import sympy as sp
    if len(text)>256:raise ValueError('Expression too long')
    tree=ast.parse(text,mode='eval')
    if sum(1 for _ in ast.walk(tree))>96:raise ValueError('Expression too complex')
    names={'pi':sp.pi,'e':sp.E}
    if allow_x:names['x']=sp.Symbol('x')
    functions={'sin':sp.sin,'cos':sp.cos,'tan':sp.tan,'sqrt':sp.sqrt,'log':sp.log,'exp':sp.exp}
    def visit(node,depth=0):
        if depth>16:raise ValueError('Expression too deep')
        child=lambda x:visit(x,depth+1)
        if isinstance(node,ast.Expression):return child(node.body)
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            if not math.isfinite(node.value) or abs(node.value)>1e12:raise ValueError('Number out of range')
            return sp.Integer(node.value) if isinstance(node.value,int) else sp.Rational(str(node.value))
        if isinstance(node,ast.Name) and node.id in names:return names[node.id]
        if isinstance(node,ast.UnaryOp):
            a=child(node.operand)
            if isinstance(node.op,ast.UAdd):return a
            if isinstance(node.op,ast.USub):return -a
        if isinstance(node,ast.BinOp):
            a,b=child(node.left),child(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Div):
                if b==0:raise ValueError('Division by zero')
                return a/b
            if isinstance(node.op,ast.Pow):
                # No symbolic exponent or tower such as 2**(2**100).
                if not b.is_number or not b.is_real or abs(float(b))>16:raise ValueError('Exponent out of range')
                if a.is_number and abs(complex(a.evalf()))>1e12:raise ValueError('Power base too large')
                return a**b
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in functions:
            if len(node.args)!=1 or node.keywords:raise ValueError('Only single-argument functions')
            arg=child(node.args[0])
            if arg.is_number and abs(complex(arg.evalf()))>10000:raise ValueError('Function argument too large')
            return functions[node.func.id](arg)
        raise ValueError('Unsupported syntax or identifier')
    result=visit(tree)
    if result.has(sp.zoo,sp.oo,-sp.oo,sp.nan):raise ValueError('Non-finite result')
    return result

def compute_math(name,arguments):
    import sympy as sp
    if name=='math.calculate':
        value=safe_expression(arguments['expression'])
        return {'exact':str(value),'numeric':str(sp.N(value,15))}
    if name=='math.differentiate':
        value=safe_expression(arguments['expression'],allow_x=True)
        return {'derivative':str(sp.diff(value,sp.Symbol('x')))}
    if name=='math.eigenvalues':
        rows=arguments['matrix'];n=len(rows)
        if not 1<=n<=4 or any(len(row)!=n for row in rows):raise ValueError('Matrix must be square, at most 4x4')
        values=[]
        for row in rows:
            if any(type(x) not in (int,float) or not math.isfinite(x) or abs(x)>1e6 for x in row):
                raise ValueError('Only bounded numeric matrix entries')
            values.append([sp.Rational(str(x)) for x in row])
        eigen=sp.Matrix(values).eigenvals()
        return {'eigenvalues':[{'value':str(k),'multiplicity':int(v)} for k,v in sorted(eigen.items(),key=lambda kv:str(kv[0]))]}
    raise ValueError('Unregistered mathematical operation')

def _worker(connection,name,arguments):
    try:
        if sys.platform.startswith('linux'):
            import resource
            resource.setrlimit(resource.RLIMIT_CPU,(5,5))
            # Address-space quotas depend on loaded libraries; use an OS/container limit in production.
            resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
        result=compute_math(name,arguments)
        connection.send({'ok':True,'result':result})
    except Exception as exc:
        connection.send({'ok':False,'error':f'{type(exc).__name__}: {str(exc)[:300]}'})
    finally:connection.close()

class ToolExecutor:
    def __init__(self,rag_path=None,timeout=8.0):
        self.rag_path,self.timeout=rag_path,timeout
        self.schemas={s['name']:s for s in SCHEMAS}
    def execute(self,call):
        name,args=call.get('name'),call.get('arguments')
        if name not in self.schemas:return {'ok':False,'error':'Unknown tool; execution denied'}
        try:jsonschema.validate(args,self.schemas[name]['parameters'])
        except jsonschema.ValidationError as exc:return {'ok':False,'error':str(exc.message)[:300]}
        if name=='retrieve':
            if not self.rag_path:return {'ok':False,'error':'No local retrieval index configured'}
            from .rag import search
            try:return {'ok':True,'result':search(self.rag_path,**args)}
            except Exception as exc:return {'ok':False,'error':str(exc)[:300]}
        ctx=mp.get_context('spawn');parent,child=ctx.Pipe(duplex=False)
        process=ctx.Process(target=_worker,args=(child,name,args),daemon=True)
        process.start();child.close()
        try:
            if parent.poll(self.timeout):
                try:return parent.recv()
                except EOFError:return {'ok':False,'error':'Tool worker exited without a result'}
            return {'ok':False,'error':'Tool execution timed out'}
        finally:
            parent.close()
            process.join(timeout=.2)
            if process.is_alive():process.terminate();process.join(timeout=1)
            if process.is_alive():process.kill();process.join()
