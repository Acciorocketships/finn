import torch
from finn import Finn

def run():
        torch.manual_seed(3)

        x = torch.tensor([[-0.0623,  0.9896,  0.2445,  1.7579, -0.4431,  0.8229],
                [-0.4769,  0.8067, -0.1701,  1.5750, -0.8577,  0.6400],
                [ 0.1788, -0.6471,  0.7415, -0.0595,  0.2898, -0.3654],
                [-0.1804, -0.7816,  0.3824, -0.1941, -0.0693, -0.5000],
                [-0.1520, -0.0546, -0.8480,  0.0289,  0.4494, -0.6795],
                [-0.2676,  0.3443, -0.9635,  0.4278,  0.3339, -0.2806],
                [-0.6990, -0.3817, -1.4810, -0.6112, -0.6195, -1.3507],
                [-0.9782,  0.9249, -1.7602,  0.6955, -0.8987, -0.0440],
                [-0.9344,  0.8774, -1.0204,  0.3389, -1.1981,  0.9061],
                [-0.1763, -0.4208, -0.2623, -0.9593, -0.4400, -0.3921],
                [ 0.6577,  0.8560, -0.0371,  1.0046,  1.4251, -0.0175],
                [ 0.3061, -0.2086, -0.3887, -0.0600,  1.0735, -1.0821],
                [-0.1189, -0.0273, -0.1115, -0.5411,  0.8131, -0.1608],
                [-0.4061, -0.3701, -0.3986, -0.8840,  0.5260, -0.5037],
                [-0.3714, -0.3466, -0.6782,  0.5443, -0.7456,  0.0254],
                [-0.0426,  0.5480, -0.3494,  1.4389, -0.4168,  0.9200],
                [-0.2581, -0.6179, -0.2929, -0.8867,  0.2894, -1.3510],
                [-0.0048, -0.1900, -0.0396, -0.4588,  0.5427, -0.9232],
                [-0.5454,  0.7155, -0.5726, -0.2553, -0.4613,  1.2852],
                [-0.9955,  0.7582, -1.0227, -0.2126, -0.9114,  1.3278],
                [-0.2962, -0.0689,  0.6554,  0.9208, -0.5734, -0.1020],
                [ 0.7267,  0.2282,  1.6783,  1.2179,  0.4495,  0.1951],
                [-0.8584,  0.9418,  0.0158,  1.4609, -0.9987,  0.7170],
                [ 0.8078, -0.1545,  1.6820,  0.3646,  0.6675, -0.3793],
                [ 0.7483,  0.5703,  0.6652, -0.0699,  0.7375,  1.5013],
                [ 0.6818, -0.5776,  0.5987, -1.2178,  0.6710,  0.3535],
                [ 0.4083,  0.8753, -0.0866,  1.0317,  0.2773,  1.7980],
                [-0.3284, -0.9157, -0.8233, -0.7593, -0.4595,  0.0070],
                [-0.5043,  0.7909, -1.1761,  1.3240, -0.7519,  0.7419],
                [ 0.5226, -0.2917, -0.1492,  0.2413,  0.2750, -0.3408],
                [-0.3413, -0.6498, -0.3847, -1.2599, -0.2518, -0.3909],
                [-0.9369, -0.9260, -0.9803, -1.5361, -0.8474, -0.6671],
                [ 0.4486,  0.2927, -0.4088,  1.1394,  0.3010, -0.3557],
                [-0.2474, -0.1853, -1.1048,  0.6614, -0.3950, -0.8336],
                [-0.3703, -0.4147, -0.7885, -0.4538, -0.7370, -0.7511],
                [-0.2034,  0.0163, -0.6216, -0.0228, -0.5701, -0.3201],
                [ 0.2637,  0.5018,  1.1269,  0.8055, -0.4185,  0.1010],
                [-0.3531, -0.0644,  0.5101,  0.2393, -1.0354, -0.4652],
                [ 0.2753, -0.0899,  1.1543, -0.8433, -0.2231, -0.5298],
                [ 0.4200, -0.4912,  1.2989, -1.2447, -0.0784, -0.9311],
                [-0.6295,  0.0866,  0.1459, -0.4681, -0.2129,  0.3105],
                [-0.4557,  0.6941,  0.3197,  0.1394, -0.0391,  0.9179],
                [ 0.2505, -0.6916,  0.2333, -0.8708,  1.1478, -0.6275],
                [-0.1159,  0.5632, -0.1331,  0.3840,  0.7814,  0.6272],
                [ 0.0612, -0.8937,  0.6935, -1.0463,  0.4599, -1.6960],
                [-0.2888, -0.2464,  0.3436, -0.3990,  0.1099, -1.0486],
                [ 0.8734,  0.3766,  1.1934, -0.1405,  0.2638,  1.1826],
                [-0.9391, -0.3951, -0.6191, -0.9122, -1.5487,  0.4109],
                [-0.3618,  0.2324, -0.6135,  0.5136,  0.5857,  0.9236],
                [-0.1651,  0.6287, -0.4168,  0.9099,  0.7824,  1.3199],
                [-0.2571,  0.0869, -0.6472,  0.4971,  0.6389,  0.0137],
                [-0.8475,  0.8700, -1.2376,  1.2802,  0.0485,  0.7969],
                [ 0.8616,  0.6792,  0.6297,  0.7888,  0.2826,  1.0753],
                [ 0.5831, -0.9156,  0.3512, -0.8061,  0.0041, -0.5195],
                [-0.8166, -0.4176, -1.7443, -0.8566, -0.2261, -0.4748],
                [ 0.5345, -0.2161, -0.3932, -0.6551,  1.1250, -0.2733],
                [ 0.0722,  0.3550, -0.8297,  0.8396, -0.6586, -0.5284],
                [ 0.9368, -0.0928,  0.0348,  0.3918,  0.2059, -0.9762],
                [ 0.9635, -0.5508,  1.6015, -1.4858,  0.6519, -0.4348],
                [-0.3358, -0.3621,  0.3022, -1.2971, -0.6474, -0.2461],
                [-0.4484,  0.2072, -0.4452,  0.8922, -0.6994,  1.0598],
                [ 0.0509,  0.8537,  0.0540,  1.5387, -0.2002,  1.7063],
                [ 0.9587, -0.2283,  0.7935, -0.6897,  0.1643,  0.2263],
                [ 0.1959, -0.1477,  0.0308, -0.6092, -0.5985,  0.3069],
                [ 0.8873,  0.7003,  0.0143,  1.4186,  1.8136, -0.2180],
                [-0.4000, -0.7161, -1.2730,  0.0022,  0.5263, -1.6345],
                [ 0.9303,  0.8745,  1.2403,  0.5534,  0.1980,  1.0942],
                [ 0.9166,  0.0049,  1.2265, -0.3161,  0.1843,  0.2246],
                [-0.0954,  0.8685,  0.2976,  0.1701, -0.4197,  1.3070],
                [-0.5175, -0.1085, -0.1245, -0.8069, -0.8418,  0.3300],
                [ 0.9046,  0.0450,  1.5381, -0.0756,  1.5747, -0.4051],
                [-0.8509, -0.5833, -0.2174, -0.7039, -0.1808, -1.0334],
                [-0.4357,  0.5897, -1.1726,  0.6899, -0.6790,  0.4659],
                [-0.9013, -0.3325, -1.6381, -0.2323, -1.1446, -0.4563],
                [-0.3423, -0.9991, -0.0590, -1.6351, -0.0048, -1.9156],
                [-0.4816,  0.3222, -0.1983, -0.3137, -0.1441, -0.5942],
                [ 0.3782, -0.5866,  0.7528,  0.1314,  0.7796, -1.4194],
                [ 0.8408,  0.4423,  1.2153,  1.1604,  1.2422, -0.3904],
                [ 0.2112,  0.1787, -0.4651,  0.8534,  0.3196, -0.3844],
                [-0.3264,  0.1744, -1.0028,  0.8491, -0.2181, -0.3887],
                [ 0.5513,  0.0056, -0.3689,  0.8579,  1.0227, -0.8636],
                [ 0.9649,  0.9528,  0.0447,  1.8051,  1.4363,  0.0837],
                [ 0.9503, -0.7851,  0.9925, -0.1695,  0.0759, -1.2227],
                [-0.4595,  0.1601, -0.4173,  0.7756, -1.3339, -0.2775],
                [-0.6260, -0.4426,  0.0789, -0.4686, -1.4867, -0.1055],
                [ 0.0145,  0.3834,  0.7194,  0.3575, -0.8462,  0.7206],
                [ 0.5307, -0.4671, -0.2841, -0.0136,  0.6374, -0.1406],
                [-0.6198,  0.9007, -1.4346,  1.3541, -0.5131,  1.2271],
                [ 0.3346, -0.9908,  0.6766, -1.2135,  1.2213, -1.4754],
                [ 0.9765, -0.1739,  1.3185, -0.3967,  1.8633, -0.6585],
                [ 0.5276,  0.0954,  0.9884,  0.0616,  0.5887,  0.2892],
                [ 0.8022,  0.7939,  1.2630,  0.7602,  0.8633,  0.9878],
                [ 0.4742,  0.6817,  1.1772, -0.1506,  0.4081,  0.4856],
                [ 0.4329,  0.0165,  1.1360, -0.8158,  0.3669, -0.1796],
                [-0.3741,  0.4059,  0.1069,  0.8395, -1.0425,  1.3749],
                [-0.5463, -0.7488, -0.0653, -0.3152, -1.2147,  0.2201],
                [-0.7904, -0.4751, -0.2180,  0.4727, -0.8225,  0.0192],
                [ 0.7878, -0.5486,  1.3602,  0.3992,  0.7556, -0.0543],
                [ 0.8004, -0.9183,  1.3776, -1.5797,  1.3851, -0.9803],
                [ 0.0910,  0.3722,  0.6682, -0.2892,  0.6757,  0.3102],
                [-0.5702, -0.8068,  0.3407, -0.5539, -0.8356, -0.7575],
                [ 0.6612, -0.4327,  1.5722, -0.1799,  0.3959, -0.3835],
                [ 0.6393,  0.0185,  0.8409, -0.9388,  1.2067,  0.4993],
                [ 0.4174,  0.2405,  0.6189, -0.7169,  0.9848,  0.7213],
                [-0.9526,  0.6879, -1.3547,  1.0851, -0.3324,  1.5602],
                [ 0.5310,  0.9843,  0.1289,  1.3814,  1.1513,  1.8565],
                [-0.1730, -0.4050,  0.5861, -0.3940,  0.7208, -1.0405],
                [-0.8775,  0.3397, -0.1184,  0.3506,  0.0163, -0.2959],
                [ 0.0106, -0.5441,  0.0120, -0.3072, -0.7816,  0.2860],
                [-0.5545, -0.0936, -0.5531,  0.1433, -1.3466,  0.7366],
                [ 0.1730, -0.5335,  0.5418, -1.0000, -0.1367,  0.4491],
                [ 0.5998,  0.2500,  0.9685, -0.2166,  0.2900,  1.2325],
                [ 0.0808, -0.0866, -0.4029, -0.4697,  0.1519, -1.0514],
                [-0.2083, -0.5761, -0.6920, -0.9593, -0.1372, -1.5409],
                [ 0.3151, -0.9497, -0.5510, -0.4171, -0.6556, -1.4363],
                [-0.8700, -0.8403, -1.7361, -0.3077, -1.8406, -1.3269],
                [-0.1113, -0.8824,  0.0212, -1.5414,  0.0794, -0.2843],
                [ 0.9094, -0.7517,  1.0420, -1.4106,  1.1002, -0.1536],
                [ 0.2449,  0.9844,  0.7482,  1.0479,  0.8974,  1.7142],
                [-0.0216, -0.8391,  0.4817, -0.7756,  0.6309, -0.1093]])

        x_lim_lower = torch.tensor([-1.,-1.,-1.,-1.,-1.,-1.]) * 1
        x_lim_upper = torch.tensor([1.,1.,1.,1.,1.,1.]) * 1

        x = torch.clamp(x, x_lim_lower, x_lim_upper)

        area = torch.prod(x_lim_upper-x_lim_lower)

        f = Finn(
                dim=x.shape[-1],
                area=area,
                pos=True,
                k=1,
                nlayers=3,
                x_lim_lower=x_lim_lower,
                x_lim_upper=x_lim_upper,
        )

        y = f(x)
        print(y)

if __name__ == "__main__":
        run()