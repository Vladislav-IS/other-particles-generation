N = 2112  # neutron PDG
P = 2212  # proton PDG
pion_masses = {211: 0.13957039, -211: 0.13957039, 111: 0.1349768}  # pions masses
pions = list(pion_masses)  # pions PDG
directions = ["x", "y", "z"]  # momenta components
nucleons = [N, P]
particles = nucleons + pions
label_to_num = {col: i for i, col in enumerate(pions)}  # ordinal encoding for pions PDG
num_to_label = {v: k for k, v in label_to_num.items()}  # ordinal decoding for pions PDG
feat_dim = 3  # generated data dimensionality
seeds = [0xF00D, 0xC0FFEE, 0xDEAD]  # random seeds for experiments
part_names = {
    111: r"$\pi_0$",
    -211: r"$\pi^-$",
    211: r"$\pi^+$",
    2112: r"$n$",
    2212: r"$p$",
}
