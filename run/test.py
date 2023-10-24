import finn
import torch
import time

if __name__ == "__main__":
    init_start_time = time.time()

    dim = 10

    finn = finn.Finn(
        dim=dim,
        area=0.1,
        nlayers=3,
        pos=True,
        x_lim_lower=-1 * torch.ones(dim),
        x_lim_upper=1 * torch.ones(dim),
        condition=True,
    )

    data = torch.randn(200, dim)
    out = finn(data)
    data = torch.randn(200, dim)
    out = finn(data)
    loss = torch.mean(out)
    loss.backward()