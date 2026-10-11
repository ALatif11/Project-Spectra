"""Explicit staged CLI: metadata, selected download, prepare, benchmark, compare."""
import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from src import inthewild as itw


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=['inspect','download','prepare','extract','compare'])
    p.add_argument('--root',type=Path,default=Path('data/inthewild'))
    p.add_argument('--experiment',type=Path,default=Path('data/inthewild/experiment-12000-v1'))
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--samples',type=int,default=12000)
    p.add_argument('--limit',type=int)
    p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--baseline',type=Path,default=Path('models/baseline.joblib'))
    p.add_argument('--model',type=Path,default=Path('models/inthewild-v1-12000.joblib'))
    p.add_argument('--real',type=Path,action='append',default=[])
    args=p.parse_args()
    try:
        with threadpool_limits(limits=1):
            if args.stage=='inspect': result=itw.inspect(args.root)
            elif args.stage=='download': result=itw.download(args.root,args.shards,min(args.seconds,600))
            elif args.stage=='prepare': result=itw.prepare(args.root,args.experiment,args.samples)
            elif args.stage=='extract':
                result=itw.extract(args.root,args.experiment,args.limit,min(args.seconds,900))
                name='benchmark.json' if args.limit else 'extraction.json'
                itw.save_json(args.experiment/name,result)
            else: result=itw.compare(args.experiment,args.baseline,args.model,[(path,0) for path in args.real])
    except (ValueError,OSError,RuntimeError) as exc:
        p.exit(1,f'{exc}\n')
    if args.stage in ('inspect','prepare'): result={k:v for k,v in result.items() if k not in ('distribution','available','selected','partitions')}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
