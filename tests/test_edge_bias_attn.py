import torch
from tcdfkg.anomaly.edge_bias_attn import EdgeBiasAttention

def test_bias_prefers_higher_score_and_smaller_lag():
    tcg = {"edges":[
        {"src":0, "dst":2, "lag":2, "score":0.9},  # tốt: score cao, lag nhỏ
        {"src":1, "dst":2, "lag":8, "score":0.3},  # kém: score thấp, lag lớn
    ]}
    attn = EdgeBiasAttention(tcg, dmax=16, mlp_hidden=8, device="cpu")

    B, N, C = 4, 3, 16
    Q = torch.randn(B,N,C)
    K = torch.randn(B,N,C)
    V = torch.randn(B,N,C)

    # inject structure: make parent 0 more similar to target 2 than parent 1
    Q[:,2,:] = K[:,0,:] * 0.8 + 0.1
    Q[:,2,:] += torch.randn_like(Q[:,2,:])*0.01

    H = attn(Q,K,V)
    # soft check: message to node 2 should align more with V[:,0,:] than V[:,1,:]
    # compute alpha indirectly via cosine similarities between H[:,2,:] and V parents
    import torch.nn.functional as F
    cos0 = F.cosine_similarity(H[:,2,:], V[:,0,:], dim=-1).mean().item()
    cos1 = F.cosine_similarity(H[:,2,:], V[:,1,:], dim=-1).mean().item()
    assert cos0 > cos1, "Edge-bias should increase contribution from (0->2) vs (1->2)"
