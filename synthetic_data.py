import numpy as np

MTU = 1500
CLASS_NAMES = ["chat", "email", "streaming", "voip"]

# length/byte/timing are each Gaussian(mean, std) clipped to a shared range,
# so classes overlap instead of occupying disjoint bands.
_LEN_BOUNDS = (64, 1500)
_CLASS_PARAMS = {
    #                 len_mean  len_std  byte_mean  byte_std  iat_mean  iat_std
    "chat":      dict(len_mean=260,  len_std=190, byte_mean=100, byte_std=55, iat_mean=0.35, iat_std=0.30),
    "email":     dict(len_mean=680,  len_std=420, byte_mean=130, byte_std=65, iat_mean=0.70, iat_std=0.55),
    "streaming": dict(len_mean=1250, len_std=300, byte_mean=160, byte_std=55, iat_mean=0.12, iat_std=0.10),
    "voip":      dict(len_mean=210,  len_std=90,  byte_mean=115, byte_std=40, iat_mean=0.05, iat_std=0.02),
}


def _make_flow(rng, params, n_packets):
    lo, hi = _LEN_BOUNDS
    vecs, sizes = [], []
    for _ in range(n_packets):
        length = int(np.clip(rng.normal(params["len_mean"], params["len_std"]), lo, hi))
        # class-conditioned byte signal + independent per-packet noise, so
        # content alone is not a perfect predictor either
        payload = np.clip(
            rng.normal(params["byte_mean"], params["byte_std"], size=length)
            + rng.normal(0, 20, size=length),
            0, 255,
        ).astype(np.uint8)
        vec = np.zeros(MTU, dtype=np.float32)
        vec[:length] = payload
        vecs.append(vec / 255.0)
        sizes.append(float(length))

    iats = np.clip(rng.normal(params["iat_mean"], params["iat_std"], size=n_packets), 1e-3, None)
    timestamps = np.cumsum(iats).astype(np.float64)

    return np.stack(vecs), timestamps, np.array(sizes, dtype=np.float32)


def generate_synthetic_flows(n_flows_per_class=50, min_packets=6, max_packets=10, seed=0):
    """
    Returns a dict: class_name -> list of (flow_vectors, timestamps, sizes)
    tuples, exactly matching what flows_to_vectors() returns per class.
    """
    rng = np.random.default_rng(seed)
    out = {cls: [] for cls in CLASS_NAMES}
    for cls in CLASS_NAMES:
        params = _CLASS_PARAMS[cls]
        for _ in range(n_flows_per_class):
            n_packets = int(rng.integers(min_packets, max_packets + 1))
            out[cls].append(_make_flow(rng, params, n_packets))
    return out


if __name__ == "__main__":
    data = generate_synthetic_flows(n_flows_per_class=5, seed=0)
    for cls, flows in data.items():
        v, t, s = flows[0]
        print(f"{cls:10s} example flow: vecs={v.shape} timestamps={t.shape} "
              f"sizes mean={s.mean():.1f}  IAT mean={np.diff(t).mean():.4f}s")
