from typing import Dict, List, Tuple

def fuse_scores(x_edges: List[dict], s_kg: Dict[Tuple[int,int], float],
                lambda_mix: float = 0.5, theta_edge: float = 0.15) -> dict:
    out = {'edges': []}
    for e in x_edges:
        i, j = e['src'], e['dst']
        s = (1 - lambda_mix) * e['s_att'] + lambda_mix * s_kg.get((i, j), 0.0)
        if s >= theta_edge:
            out['edges'].append({'src': i, 'dst': j, 'lag': int(e['lag']), 'score': float(s)})
    return out
