MIN_SARLE_K = 4


def analyze(*_args, pnp_k=3, **_kwargs):
    return {"status": "not_available", "reason": f"geometry artifacts/larger-K data unavailable; Sarle requires K>={MIN_SARLE_K}, observed K={pnp_k}", "minimum_k": MIN_SARLE_K, "observed_k": pnp_k}
