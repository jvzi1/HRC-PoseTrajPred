HRC-PoseTrajPred is an advanced system for human-robot collaboration, focusing on human pose recognition and trajectory prediction to enhance cooperative interactions.

## stgcn
from stgcn import model 调用模型
"""
if args.graph_conv_type == 'cheb_graph_conv':
    model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
else:
    model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)
"""