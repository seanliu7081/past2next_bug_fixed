"""Visual proof of the SO(3) action-augmentation re-frame table, on real LIBERO-10 data.

Question: for each (mode, augment_position) setting of SO3ActionChunkAug, is the
augmented chunk equal to a single consistent rigid re-frame of the original by one
rotation Q?  A true re-frame by Q predicts

    position delta   dp  ->  Q @ dp
    rotation delta   w   ->  Q @ w        (equivalently R -> Q R Q^T)

so we pin Q to a known value, run the real module, and compare against that
prediction.  Only `conjugate + augment_position=True` matches on both channels.

Outputs (written next to this file):
    reframe_proof_light.png / reframe_proof_dark.png
    reframe_proof_residuals.csv   -- the table view of the bottom panel

Run:  /venv/oat/bin/python plot_reframe_proof.py
"""

import math
import os

import numpy as np
import torch
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import oat.tokenizer.oat.augment.so3_action_chunk_aug as M

HERE = os.path.dirname(os.path.abspath(__file__))
ZARR = "/workspace/past_action/data/libero/libero10_N500.zarr"

T = 16                 # action horizon this tokenizer ckpt was trained at
CHUNK_START = 89285    # a real LIBERO-10 chunk (episode 322), picked for a 3D-legible path
N_STATS = 4000         # chunks used for the dataset-wide residual panel
ANGLE_DEG = 25.0       # Q's angle; kept under the config's max_angle_deg: 30
AXIS = np.array([0.3, -0.2, 0.5])

CONFIGS = [
    ("conjugate",  True),
    ("conjugate",  False),
    ("left_noise", True),
    ("left_noise", False),
]
RIGID_TOL = 1e-5

# --- palette: validated categorical slots 1-2 + de-emphasis gray (emphasis form) ---
LIGHT = dict(
    surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", ink2="#52514e", muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7", raw="#898781", ref="#2a78d6", aug="#eb6834",
    band="#f0efec",
)
DARK = dict(
    surface="#1a1a19", page="#0d0d0d", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
    grid="#2c2c2a", axis="#383835", raw="#898781", ref="#3987e5", aug="#d95926",
    band="#232322",
)


# ----------------------------------------------------------------------------- data
def load_actions():
    z = zarr.open(ZARR, mode="r")
    return np.asarray(z["data/action"]).astype(np.float32)


def pin_Q():
    """Force the module's random rotation to a known Q and return it."""
    axis = AXIS / np.linalg.norm(AXIS)
    eps = torch.tensor(axis * math.radians(ANGLE_DEG), dtype=torch.float32)
    M.sample_random_rotvec = (
        lambda batch_size, device, dtype, max_angle_rad:
        eps.to(device=device, dtype=dtype).expand(batch_size, 3).clone()
    )
    return M.so3_exp_map(eps).numpy().astype(np.float64), axis


def augment(chunks, mode, augment_position):
    aug = M.SO3ActionChunkAug(
        p=1.0, max_angle_deg=ANGLE_DEG, mode=mode,
        augment_position=augment_position, rot_start=3, rot_end=6,
    ).train()
    torch.manual_seed(0)
    return aug(torch.from_numpy(chunks)).numpy()


# ------------------------------------------------------------------------- plotting
def style_3d(ax, t, xlabel, ylabel, zlabel):
    ax.set_facecolor(t["surface"])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis._axinfo["grid"].update(color=t["grid"], linewidth=0.8, linestyle="-")
        axis.line.set_color(t["axis"])
        axis.line.set_linewidth(0.8)
    ax.tick_params(colors=t["muted"], labelsize=7, pad=0)
    ax.locator_params(nbins=4)
    for lbl, setter in ((xlabel, ax.set_xlabel), (ylabel, ax.set_ylabel), (zlabel, ax.set_zlabel)):
        setter(lbl, color=t["muted"], fontsize=8, labelpad=-4)
    ax.set_box_aspect((1, 1, 0.86), zoom=1.12)
    ax.view_init(elev=22, azim=-52)


def draw_path(ax, pts, color, t, lw=2.0, ls="-", z=3, wide=False, alpha=1.0):
    """2px line, round cap; endpoint marker >= 8px carrying a 2px surface ring."""
    if wide:  # the reference band the augmented line should sit inside
        ax.plot(*pts.T, color=color, lw=6.0, alpha=0.30, solid_capstyle="round", zorder=z - 1)
    ax.plot(*pts.T, color=color, lw=lw, ls=ls, alpha=alpha,
            solid_capstyle="round", dash_capstyle="round", zorder=z)
    ax.plot(*pts[-1:].T, marker="o", ms=8, mfc=color, mec=t["surface"], mew=2.0,
            ls="none", zorder=z + 1)


def build(theme_name, t, A, Q, axis_unit, one, res):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.facecolor": t["page"],
        "savefig.facecolor": t["page"],
    })
    fig = plt.figure(figsize=(21.5, 15.8))
    # two grids: the 3D panels need a narrow left margin, the residual panel a wide
    # one for its full config labels
    gs = GridSpec(2, 4, figure=fig, hspace=0.20, wspace=0.05,
                  left=0.052, right=0.986, top=0.845, bottom=0.275)
    gs_res = GridSpec(1, 1, figure=fig, left=0.215, right=0.855,
                      top=0.222, bottom=0.062)

    raw = one["raw"]
    p_raw, w_raw = raw[:, :3].astype(np.float64), raw[:, 3:6].astype(np.float64)
    # cumsum is linear, so the reference path is exactly the rotated raw path
    path_raw = np.cumsum(p_raw, 0)
    path_ref = path_raw @ Q.T
    w_ref = w_raw @ Q.T

    for col, (mode, ap) in enumerate(CONFIGS):
        out = one[(mode, ap)]
        path_aug = np.cumsum(out[:, :3].astype(np.float64), 0)
        w_aug = out[:, 3:6].astype(np.float64)
        rigid = res[(mode, ap)]["pos"] < RIGID_TOL and res[(mode, ap)]["rot"] < RIGID_TOL

        head = f"{mode}  ·  augment_position={ap}"
        ax = fig.add_subplot(gs[0, col], projection="3d")
        ax.set_title(head, color=t["ink"], fontsize=12.5,
                     fontweight="bold" if rigid else "normal", pad=10)
        for pts, c, lw, ls, wide in (
            (path_raw, t["raw"], 1.8, "-", False),
            (path_ref, t["ref"], 2.0, "-", True),
            (path_aug, t["aug"], 2.0, (0, (4, 3)), False),
        ):
            draw_path(ax, pts, c, t, lw=lw, ls=ls, wide=wide)
        style_3d(ax, t, "x", "y", "z")
        ax.text2D(0.5, -0.055, _verdict(res[(mode, ap)]["pos"], t)[0],
                  transform=ax.transAxes, ha="center", fontsize=9.5,
                  color=_verdict(res[(mode, ap)]["pos"], t)[1])

        ax = fig.add_subplot(gs[1, col], projection="3d")
        for pts, c, lw, ls, wide in (
            (w_raw, t["raw"], 1.8, "-", False),
            (w_ref, t["ref"], 2.0, "-", True),
            (w_aug, t["aug"], 2.0, (0, (4, 3)), False),
        ):
            draw_path(ax, pts, c, t, lw=lw, ls=ls, wide=wide)
        style_3d(ax, t, "ωx", "ωy", "ωz")
        ax.text2D(0.5, -0.055, _verdict(res[(mode, ap)]["rot"], t)[0],
                  transform=ax.transAxes, ha="center", fontsize=9.5,
                  color=_verdict(res[(mode, ap)]["rot"], t)[1])

    # row labels
    fig.text(0.016, 0.705, "Position channel   dims 0:3\nintegrated Δposition",
             rotation=90, va="center", ha="center", color=t["ink2"], fontsize=12)
    fig.text(0.016, 0.418, "Rotation channel   dims 3:6\nrotation vector ω (rad)",
             rotation=90, va="center", ha="center", color=t["ink2"], fontsize=12)

    _residual_panel(fig, gs_res, t, res)
    _chrome(fig, t, axis_unit)
    out_png = os.path.join(HERE, f"reframe_proof_{theme_name}.png")
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return out_png


def _verdict(err, t):
    if err < RIGID_TOL:
        return f"matches re-frame   err {err:.1e}", t["ink2"]
    return f"deviates   err {err:.1e}", t["ink"]


def _residual_panel(fig, gs, t, res):
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(t["surface"])
    labels = [f"{m}  ·  augment_position={a}" for m, a in CONFIGS]
    ys = np.arange(len(CONFIGS))[::-1]

    for y, (mode, ap) in zip(ys, CONFIGS):
        r = res[(mode, ap)]
        if r["pos"] < RIGID_TOL and r["rot"] < RIGID_TOL:      # emphasis band
            ax.axhspan(y - 0.44, y + 0.44, color=t["band"], zorder=0)
    ax.axvline(RIGID_TOL, color=t["axis"], lw=1.0, zorder=1)

    # label to the RIGHT of each marker, nudged +/-9pt so the two channels never
    # collide with each other or with the neighbouring row's labels
    for y, (mode, ap) in zip(ys, CONFIGS):
        r = res[(mode, ap)]
        for key, mk, dy in (("pos", "o", 9), ("rot", "^", -9)):
            ax.plot(r[key], y, marker=mk, ms=11, mfc=t["aug"], mec=t["surface"],
                    mew=2.0, ls="none", zorder=4)
            ax.annotate(f"{r[key]:.1e}", (r[key], y), textcoords="offset points",
                        xytext=(13, dy), ha="left", va="center",
                        fontsize=9.5, color=t["ink2"])

    ax.set_yticks(ys, labels, color=t["ink"], fontsize=11.5)
    ax.set_xscale("log")
    ax.set_xlim(4e-9, 4e0)
    ax.set_ylim(-0.62, len(CONFIGS) - 0.38)
    ax.set_xlabel(f"max |augmented − re-frame prediction|   over {N_STATS:,} real LIBERO-10 chunks   (log scale)",
                  color=t["ink2"], fontsize=11, labelpad=8)
    ax.tick_params(axis="x", colors=t["muted"], labelsize=9.5)
    ax.tick_params(axis="y", length=0, colors=t["ink"])
    ax.xaxis.grid(True, color=t["grid"], lw=0.8, ls="-")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(0.8)

    ax.text(RIGID_TOL * 1.6, len(CONFIGS) - 0.52, "float32 agreement threshold",
            color=t["muted"], fontsize=9.5, va="top")
    trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for y, (mode, ap) in zip(ys, CONFIGS):
        r = res[(mode, ap)]
        ok = r["pos"] < RIGID_TOL and r["rot"] < RIGID_TOL
        ax.text(1.025, y, "✓  rigid re-frame" if ok else "✗  not a re-frame",
                transform=trans, ha="left", va="center", fontsize=11,
                fontweight="bold" if ok else "normal", color=t["ink"] if ok else t["ink2"])
    ax.legend(handles=[
        Line2D([], [], marker="o", ms=9, mfc=t["aug"], mec=t["surface"], mew=1.6,
               ls="none", label="position channel"),
        Line2D([], [], marker="^", ms=9, mfc=t["aug"], mec=t["surface"], mew=1.6,
               ls="none", label="rotation channel"),
    ], loc="lower left", frameon=False, fontsize=10.5, labelcolor=t["ink2"], ncol=1,
        bbox_to_anchor=(0.30, 0.02))


def _chrome(fig, t, axis_unit):
    fig.text(0.052, 0.977,
             "Only conjugate + augment_position=True is a consistent rigid-frame transform",
             color=t["ink"], fontsize=22, fontweight="bold", va="top")
    ax_s = ", ".join(f"{v:+.3f}" for v in axis_unit)
    fig.text(0.052, 0.939,
             f"SO3ActionChunkAug applied to one real LIBERO-10 chunk (episode 322, steps "
             f"{CHUNK_START}–{CHUNK_START + T}, horizon {T}), with the module's random rotation pinned to a\n"
             f"known Q = {ANGLE_DEG:.0f}° about axis ({ax_s}). A true re-frame by Q sends Δposition → Q·Δp and the "
             "rotation vector → Q·ω, so where the augmented\noutput (dashed) rides inside the blue reference band, "
             "the code reproduced that prediction.",
             color=t["ink2"], fontsize=12, va="top", linespacing=1.6)

    fig.legend(handles=[
        Line2D([], [], color=t["raw"], lw=1.8, label="original chunk"),
        Line2D([], [], color=t["ref"], lw=5, alpha=0.45, label="reference: exact re-frame by Q"),
        Line2D([], [], color=t["aug"], lw=2, ls=(0, (4, 3)), label="augmented: SO3ActionChunkAug output"),
    ], loc="upper right", bbox_to_anchor=(0.986, 0.982), frameon=False,
        fontsize=12, labelcolor=t["ink2"], ncol=1, handlelength=2.8)


# ------------------------------------------------------------------------------ main
def main():
    A = load_actions()
    Q, axis_unit = pin_Q()

    one_raw = A[CHUNK_START:CHUNK_START + T][None]                      # [1, T, 7]
    rng = np.random.default_rng(0)
    many = np.stack([A[s:s + T] for s in rng.integers(0, len(A) - T, size=N_STATS)])

    p, w = many[..., :3].astype(np.float64), many[..., 3:6].astype(np.float64)
    ref_p, ref_w = p @ Q.T, w @ Q.T

    one, res = {"raw": one_raw[0]}, {}
    for mode, ap in CONFIGS:
        one[(mode, ap)] = augment(one_raw, mode, ap)[0]
        out = augment(many, mode, ap)
        res[(mode, ap)] = {
            "pos": float(np.abs(out[..., :3].astype(np.float64) - ref_p).max()),
            "rot": float(np.abs(out[..., 3:6].astype(np.float64) - ref_w).max()),
        }

    paths = [build(n, t, A, Q, axis_unit, one, res)
             for n, t in (("light", LIGHT), ("dark", DARK))]

    csv = os.path.join(HERE, "reframe_proof_residuals.csv")
    with open(csv, "w") as f:
        f.write("mode,augment_position,max_pos_residual,max_rot_residual,verdict\n")
        for mode, ap in CONFIGS:
            r = res[(mode, ap)]
            ok = r["pos"] < RIGID_TOL and r["rot"] < RIGID_TOL
            f.write(f"{mode},{ap},{r['pos']:.3e},{r['rot']:.3e},"
                    f"{'rigid re-frame' if ok else 'not a re-frame'}\n")

    for pth in paths + [csv]:
        print("wrote", pth)
    for mode, ap in CONFIGS:
        r = res[(mode, ap)]
        print(f"  {mode:11s} pos={ap!s:5s}  pos {r['pos']:.2e}  rot {r['rot']:.2e}")


if __name__ == "__main__":
    main()
