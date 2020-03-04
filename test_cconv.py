import torch
import torch.nn as nn
import torch.optim as optim
import fastpatch as fp
import cconv



# Feature 4 x 3 x 1
feat = torch.tensor(
    list(range(12)), dtype=torch.float32).cuda().view(4, 3, 1)

# Model
points_cconfig = cconv.CConvConfig(3, 2)  # Spatial=3, MaxSize=2
# Share the same cconv config (cconfig)
model = nn.Sequential(
    cconv.CConvFixed(3, 4, points_cconfig),
    nn.ReLU(),
    cconv.CConv(4, 1, points_cconfig))
model.cuda()


optimizer = optim.SGD(model.parameters(), lr=0.0001)
model.train()

# Find neighbors in the scene and update fp and config
# Graph
# 0: 1, 3
# 1: 0
# 2:
# 3: 0
nn_offset = torch.tensor([0, 2, 3, 3, 4], dtype=torch.int32).cuda()
nn_list = torch.tensor([1, 3, 0, 0], dtype=torch.int32).cuda()
nw_list = torch.tensor(list(range(4 * 3)), dtype=torch.float32).cuda()
grad_nn_offset = torch.tensor([0, 2, 3, 3, 4], dtype=torch.int32).cuda()
grad_nn_list = torch.tensor([1, 0, 3, 0, 0, 0, 0, 1], dtype=torch.int32).cuda()

fp.update_feat_config(2, nn_offset, nn_list, grad_nn_offset, grad_nn_list)
fp.update_fixed_config(2, nn_offset, nn_list)
points_cconfig.update(nn_offset, nw_list)
# Iterate
for i in range(10):
    optimizer.zero_grad()

    y = model(feat)
    loss_1 = 0.8 * torch.norm(y)
    print("Step", i, "Loss_1", loss_1)
    loss_1.backward()

    y = model(feat)
    loss_2 = 0.2 * torch.norm(y)
    print("Step", i, "Loss_2", loss_2)  # 0.25 * loss_1
    loss_2.backward()

    optimizer.step()
