"""Instance extractor (paper Sec. III-D) and panoptic post-processing."""

import torch


@torch.no_grad()
def extract_instances(sem, mask, is_thing, xyz, offset, offset_err,
                      embedding, embedding_err, features="both",
                      learned_radius=True, center_radius=0.5,
                      embedding_radius=0.5, scale=(1.0, 1.0), min_points=1,
                      radius_mode="seed", generator=None):
    """Iteratively grow instances from random seeds on one range image.

    A random unassigned thing pixel is picked as seed; every unassigned pixel
    of the same predicted class whose predicted centre (x - o) and/or
    embedding lies within the seed's radius joins its instance. Repeat until
    all thing pixels are assigned. Shapes: sem/mask (H, W); xyz/offset
    (3, H, W); embedding (E, H, W); *_err (1, H, W).
    With ``radius_mode="seed"`` (paper) a pixel joins when it is within the
    seed's radius. Two noisy centre predictions with errors e1, e2 can be up
    to e1 + e2 apart, so ``"sum"`` compares against the seed's plus the
    candidate's radius instead (only matters for learned radii).
    Returns instance ids (H, W), 0 for stuff and empty pixels.
    """
    assert radius_mode in ("seed", "sum")
    H, W = sem.shape
    inst = torch.zeros(H * W, dtype=torch.long, device=sem.device)
    thing = (is_thing[sem] & mask).flatten()
    pix = thing.nonzero().squeeze(1)
    if pix.numel() == 0:
        return inst.view(H, W)

    use_center = features in ("center", "both")
    use_embedding = features in ("embedding", "both")
    cls = sem.flatten()[pix]
    centers = (xyz - offset).flatten(1)[:, pix].T.float()
    emb = embedding.flatten(1)[:, pix].T.float()
    if learned_radius:
        r_c = offset_err.flatten()[pix].float() * scale[0]
        r_e = embedding_err.flatten()[pix].float() * scale[1]
    else:
        r_c = torch.full_like(cls, center_radius, dtype=torch.float)
        r_e = torch.full_like(cls, embedding_radius, dtype=torch.float)

    ids = torch.zeros(pix.numel(), dtype=torch.long, device=pix.device)
    assigned = torch.zeros(pix.numel(), dtype=torch.bool, device=pix.device)
    remaining = torch.arange(pix.numel(), device=pix.device)
    next_id = 1
    while remaining.numel() > 0:
        pick = torch.randint(remaining.numel(), (1,), generator=generator).item()
        s = remaining[pick]
        cand = remaining[cls[remaining] == cls[s]]
        member = cand == s
        near = torch.ones_like(member)
        both = radius_mode == "sum" and learned_radius
        if use_center:
            r = r_c[s] + r_c[cand] if both else r_c[s]
            near &= (centers[cand] - centers[s]).norm(dim=1) <= r
        if use_embedding:
            r = r_e[s] + r_e[cand] if both else r_e[s]
            near &= (emb[cand] - emb[s]).norm(dim=1) <= r
        members = cand[member | near]
        if members.numel() >= min_points:
            ids[members] = next_id
            next_id += 1
        assigned[members] = True
        remaining = remaining[~assigned[remaining]]

    inst[pix] = ids
    return inst.view(H, W)


@torch.no_grad()
def panoptic_inference(out, batch, is_thing, cluster_cfg, scale=(1.0, 1.0),
                       generator=None):
    """Network outputs -> per-point (semantic class, instance id) per scan.

    Labels are recovered for every original point through its (u, v) pixel
    (Sec. III-E), so points hidden behind the closest one share its labels.
    """
    sem = out["logits"].argmax(1)
    results = []
    for b in range(sem.shape[0]):
        inst = extract_instances(
            sem[b], batch["mask"][b], is_thing, batch["xyz"][b],
            out["offset"][b], out["offset_err"][b], out["embedding"][b],
            out["embedding_err"][b], features=cluster_cfg["features"],
            learned_radius=cluster_cfg["learned_radius"],
            center_radius=cluster_cfg["fixed_center_radius"],
            embedding_radius=cluster_cfg["fixed_embedding_radius"],
            scale=scale, min_points=cluster_cfg["min_points"],
            radius_mode=cluster_cfg.get("radius_mode", "seed"),
            generator=generator)
        px, py = batch["px"][b].to(sem.device), batch["py"][b].to(sem.device)
        results.append((sem[b][py, px], inst[py, px]))
    return results
