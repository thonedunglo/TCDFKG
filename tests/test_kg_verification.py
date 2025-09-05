import json, os, tempfile
from tcdfkg.kg.verification import KGVerifier

def _toy_schema():
    return {
        "machines": [{
            "name": "M1",
            "components": [{"name":"C1"}],
            "feature_types": [{"name":"A"}, {"name":"B"}]
        }],
        "rules": [{
            "name":"R1",
            "logic":[{"of_component":"C1","feature_type":"B"}]
        }],
        "events": [{
            "description":"E1",
            "rule_ref":"R1",
            "sources":[{"feature_type":"A"}]
        }]
    }

def test_kg_scores_basic():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "schema.json")
        with open(p, "w") as f: json.dump(_toy_schema(), f)
        ver = KGVerifier(p, alpha=0.5, beta=0.3, delta=0.2, kappa=0.7)

        meta = {
            "candidates":[(0,1), (1,0)],
            "type": {0:"A", 1:"B"}
        }
        scores = ver.score(meta)
        assert scores[(0,1)] > scores[(1,0)], "A -> B should have stronger KG support than B -> A"
        assert 0.0 <= scores[(0,1)] <= 1.0
