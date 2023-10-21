import finn
import torch
import time

if __name__ == "__main__":
    init_start_time = time.time()

    dim = 10

    finn = finn.Finn(
        dim=dim,
        area=0.1,
        nlayers=2,
        pos=True,
        x_lim_lower=-1 * torch.ones(dim),
        x_lim_upper=1 * torch.ones(dim),
        condition=True,
    )

    init_time = time.time() - init_start_time
    print(f"It took {init_time}s to init FINN")

    data = torch.randn(200, dim)

    forward_start_time = time.time()
    out = finn(data)
    forward_time = time.time() - forward_start_time
    print(f"It took {forward_time}s to foward FINN")