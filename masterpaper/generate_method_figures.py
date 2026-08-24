#!/usr/bin/env python3
"""Generate publication-ready vector diagrams for thesis Figures 6--9."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

INK = "#252A30"
MUTED = "#69727D"
BLUE, BLUE_D = "#DDECF7", "#568CB2"
PURPLE, PURPLE_D = "#E9DDF6", "#8C63B1"
GREEN, GREEN_D = "#DCEFD9", "#578C56"
YELLOW, YELLOW_D = "#FFF0C2", "#B38A2D"
RED, RED_D = "#F9DDDA", "#C55A52"
GREY, GREY_D = "#F1F3F5", "#969FA9"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.2,
    "text.color": INK,
    "axes.linewidth": 0.7,
    "svg.fonttype": "none",
})


def canvas(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def panel(ax, x, y, w, h, label, fill=GREY, edge=GREY_D):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.014",
        facecolor=fill, edgecolor=edge, linewidth=0.75, zorder=0,
    ))
    ax.text(x + 0.012, y + h - 0.025, label, ha="left", va="top",
            fontsize=6.4, fontweight="bold", color=edge)


def node(ax, x, y, w, h, title, detail="", fill=GREY, edge=GREY_D,
         title_size=7.2, detail_size=6.0, radius=0.012):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.005,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=0.8, zorder=3,
    ))
    ty = y + h * (0.62 if detail else 0.50)
    ax.text(x + w / 2, ty, title, ha="center", va="center",
            fontsize=title_size, fontweight="semibold", zorder=4)
    if detail:
        ax.text(x + w / 2, y + h * 0.27, detail, ha="center", va="center",
                fontsize=detail_size, color="#3F4852", linespacing=1.18, zorder=4)


def arr(ax, a, b, label="", color=INK, rad=0, lw=0.8, label_offset=0.016):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=7.5, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", zorder=2,
        shrinkA=1.5, shrinkB=1.5,
    ))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + label_offset,
                label, ha="center", va="bottom", fontsize=5.7,
                color=color, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=.9))


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.035,
                facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=360, bbox_inches="tight",
                pad_inches=0.035, facecolor="white")
    plt.close(fig)


def figure6_pipeline():
    fig, ax = canvas((12.0, 4.0))
    panel(ax, .015, .08, .155, .84, "INPUTS", "#FAFBFC", GREY_D)
    panel(ax, .185, .08, .345, .84, "LOCAL VISUAL RELOCALIZATION", BLUE, BLUE_D)
    panel(ax, .545, .08, .235, .84, "TEMPORAL ESTIMATION", PURPLE, PURPLE_D)
    panel(ax, .795, .08, .19, .84, "STATE FUSION", YELLOW, YELLOW_D)

    node(ax, .035, .65, .115, .14, "UAV frames", r"$I_{t-2}, I_{t-1}, I_t$", BLUE, BLUE_D)
    node(ax, .035, .43, .115, .14, "Coarse prior", "bounded-error\nroute centre", YELLOW, YELLOW_D)
    node(ax, .035, .21, .115, .14, "Orthomosaic", "fixed anchor lattice", GREEN, GREEN_D)

    node(ax, .205, .66, .125, .13, "Frozen backbone", "MobileCLIP2-S2", BLUE, BLUE_D)
    node(ax, .205, .43, .125, .13, "Candidate selector", "local lattice → forward set", GREEN, GREEN_D)
    node(ax, .205, .20, .125, .13, "SAT cache", "precomputed embeddings", GREEN, GREEN_D)
    node(ax, .375, .55, .13, .16, "Similarity field", "cosine scores\n+ MS", RED, RED_D)
    node(ax, .375, .27, .13, .14, "Visual output", r"$r_t^{MS},\;\Sigma_t^{MS}$", RED, RED_D)

    node(ax, .565, .64, .19, .13, "Temporal statistics", r"mean, $\Delta z_t$, $\Delta^2z_t$", PURPLE, PURPLE_D)
    node(ax, .565, .42, .19, .14, "Three-frame GRU", "visual context + previous state", PURPLE, PURPLE_D)
    node(ax, .565, .20, .09, .12, "Correction", r"$\delta r_t, q_t$", RED, RED_D)
    node(ax, .665, .20, .09, .12, "Motion", r"$v_t, a_t$", GREEN, GREEN_D)

    node(ax, .815, .61, .15, .14, "2nd-order prior", "heading-aware\ndisplacement", YELLOW, YELLOW_D)
    node(ax, .815, .38, .15, .14, "Kalman fusion", "uncertainty-aware\nconstrained update", YELLOW, YELLOW_D)
    node(ax, .815, .16, .15, .12, "Route state", r"$[s,e,v_s,v_e]_t$", GREEN, GREEN_D)

    arr(ax, (.15, .72), (.205, .725)); arr(ax, (.15, .50), (.205, .495))
    arr(ax, (.15, .28), (.205, .265)); arr(ax, (.33, .725), (.375, .63))
    arr(ax, (.33, .495), (.375, .62)); arr(ax, (.33, .265), (.375, .59))
    arr(ax, (.44, .55), (.44, .41)); arr(ax, (.505, .34), (.565, .49), "visual context")
    arr(ax, (.33, .725), (.565, .705), "three frames")
    arr(ax, (.66, .64), (.66, .56)); arr(ax, (.63, .42), (.61, .32))
    arr(ax, (.69, .42), (.71, .32)); arr(ax, (.754, .26), (.815, .68), "motion prior", PURPLE_D)
    arr(ax, (.61, .32), (.815, .45), "correction + variance", RED_D)
    arr(ax, (.505, .34), (.815, .45), "visual measurement", RED_D, -.07, label_offset=.006)
    arr(ax, (.89, .61), (.89, .52)); arr(ax, (.89, .38), (.89, .28))
    ax.text(.50, .025, "The coarse position restricts candidate selection only; visual matching remains coordinate-free.",
            ha="center", va="bottom", fontsize=6.4, color=MUTED)
    save(fig, "fig06_detailed_localization_pipeline")


def figure7_forward_selection():
    fig, ax = canvas((10.0, 3.6))
    panel(ax, .02, .08, .48, .84, "FIXED LOCAL LATTICE", "#FAFBFC", GREY_D)
    panel(ax, .525, .08, .455, .84, "CAUSAL SELECTION", GREEN, GREEN_D)

    ox, oy, sx, sy = .075, .18, .058, .105
    for r in range(6):
        for c in range(6):
            forward = r >= 3
            ax.add_patch(Rectangle(
                (ox + c * sx, oy + r * sy), sx * .84, sy * .80,
                facecolor="#8ED17E" if forward else "#E7EAED",
                edgecolor=GREEN_D if forward else GREY_D, linewidth=.58,
            ))
    cx, cy = ox + 2.42 * sx, oy + 2.62 * sy
    ax.plot(cx, cy, "o", ms=4.1, color=RED_D, zorder=5)
    arr(ax, (cx, cy + .015), (cx, oy + 6.30 * sy), r"causal heading $\theta_t$", BLUE_D,
        label_offset=.012)
    ax.annotate("current coarse centre", (cx, cy), xytext=(.105, .12),
                textcoords="axes fraction", fontsize=5.8, color=RED_D,
                arrowprops=dict(arrowstyle="-", color=RED_D, lw=.65))
    ax.text(.391, .75, "FORWARD 3×6\n18 retained centres", ha="center", va="center",
            fontsize=7.0, fontweight="semibold", color=GREEN_D)
    ax.text(.391, .36, "REAR 3×6\nnot scored", ha="center", va="center",
            fontsize=6.6, color=MUTED)

    node(ax, .555, .66, .17, .14, "Heading projection", r"$p_i=(c_i-c_t)^\top f_t$", BLUE, BLUE_D)
    node(ax, .775, .66, .17, .14, "Retain candidates", "largest forward projection", GREEN, GREEN_D)
    node(ax, .555, .31, .17, .14, "Cached SAT features", "18 selected anchors", GREEN, GREEN_D)
    node(ax, .775, .31, .17, .14, "Local visual score", "cosine + MS", RED, RED_D)
    arr(ax, (.725, .73), (.775, .73)); arr(ax, (.86, .66), (.64, .45), "indices", GREEN_D, .18)
    arr(ax, (.725, .38), (.775, .38)); arr(ax, (.50, .57), (.555, .73), "heading + centres", BLUE_D)
    ax.text(.752, .17, "Only the selected satellite embeddings enter visual matching.",
            ha="center", fontsize=6.3, color=MUTED)
    save(fig, "fig07_forward_3x6_candidate_selection")


def figure8_gru_motion():
    fig, ax = canvas((11.5, 3.7))
    panel(ax, .015, .08, .235, .84, "THREE-FRAME EVIDENCE", BLUE, BLUE_D)
    panel(ax, .265, .08, .27, .84, "RECURRENT STATE", PURPLE, PURPLE_D)
    panel(ax, .55, .08, .255, .84, "PREDICTION HEADS", "#FAFBFC", GREY_D)
    panel(ax, .82, .08, .165, .84, "FILTER INPUTS", YELLOW, YELLOW_D)

    for x, lab in [(.035, r"$z_{t-2}$"), (.105, r"$z_{t-1}$"), (.175, r"$z_t$")]:
        node(ax, x, .66, .055, .10, lab, "UAV", BLUE, BLUE_D, 7.2, 5.3)
    node(ax, .035, .39, .195, .15, "Temporal statistics", r"$\bar z_t,\;\Delta z_t,\;\Delta^2z_t$", BLUE, BLUE_D)
    node(ax, .035, .18, .195, .11, "SAT context", "local similarity field", GREEN, GREEN_D)
    for x in (.062, .132, .202): arr(ax, (x, .66), (.132, .54), lw=.7)

    node(ax, .29, .61, .22, .10, "Previous hidden state", r"$h_{t-1}$", PURPLE, PURPLE_D)
    node(ax, .29, .34, .22, .17, "GRU update", r"$h_t=\mathrm{GRU}(u_t,h_{t-1})$", PURPLE, PURPLE_D)
    node(ax, .29, .16, .22, .09, "New hidden state", r"$h_t$", PURPLE, PURPLE_D)
    arr(ax, (.40, .61), (.40, .51)); arr(ax, (.40, .34), (.40, .25))
    arr(ax, (.23, .465), (.29, .425)); arr(ax, (.23, .235), (.29, .38))

    heads = [
        (.575, .67, "Motion head", r"$v_t, a_t$", GREEN, GREEN_D),
        (.69, .67, "Heading head", r"$\delta\theta_t,\omega_t$", YELLOW, YELLOW_D),
        (.575, .36, "Correction", r"$\delta r_t$", RED, RED_D),
        (.69, .36, "Variance", r"$q_t$", RED, RED_D),
    ]
    for x, y, t, d, f, e in heads:
        node(ax, x, y, .095, .13, t, d, f, e, 6.5, 5.6)
        arr(ax, (.51, .425), (x, y + .065), lw=.7)
    node(ax, .575, .16, .21, .10, "Satellite estimate", r"$r_t^{MS},\Sigma_t^{MS}$", BLUE, BLUE_D)

    node(ax, .84, .57, .125, .18, "Motion prior",
         r"$\Delta r_t^{poly}$" + "\n" + r"$=R(\delta\theta_t)(v_t+\frac{1}{2}a_t)$",
         YELLOW, YELLOW_D, 7.0, 5.4)
    node(ax, .84, .25, .125, .18, "Visual measurement",
         r"$z_t^{meas}=r_t^{MS}+\delta r_t$" + "\n" + r"$R_t=\Sigma_t^{MS}+q_t$",
         RED, RED_D, 6.5, 5.2)
    arr(ax, (.67, .735), (.84, .66)); arr(ax, (.785, .735), (.84, .66))
    arr(ax, (.67, .425), (.84, .34)); arr(ax, (.785, .425), (.84, .34))
    arr(ax, (.785, .21), (.84, .30), lw=.7)
    ax.text(.902, .14, "to constrained Kalman fusion", ha="center", fontsize=6.1, color=MUTED)
    save(fig, "fig08_gru_inertial_prediction")


def figure9_kalman():
    fig, ax = canvas((11.0, 3.5))
    panel(ax, .015, .09, .27, .82, "FILTER INPUTS", "#FAFBFC", GREY_D)
    panel(ax, .30, .09, .39, .82, "UNCERTAINTY-AWARE UPDATE", YELLOW, YELLOW_D)
    panel(ax, .705, .09, .28, .82, "CONSTRAINED POSTERIOR", GREEN, GREEN_D)

    node(ax, .125, .62, .13, .14, "Motion prior", r"$\bar x_t,\bar P_t$", PURPLE, PURPLE_D)
    node(ax, .125, .29, .13, .14, "Visual estimate", r"$z_t^{meas},R_t$", BLUE, BLUE_D)
    node(ax, .04, .62, .065, .14, "Model", "2nd order", YELLOW, YELLOW_D, 6.5, 5.4)
    node(ax, .04, .29, .065, .14, "Variance", "MS + learned", RED, RED_D, 6.5, 5.1)
    arr(ax, (.105, .69), (.125, .69)); arr(ax, (.105, .36), (.125, .36))

    node(ax, .33, .64, .13, .13, "Innovation", r"$\nu_t=z_t-H\bar x_t$", YELLOW, YELLOW_D, 6.8, 5.5)
    node(ax, .52, .64, .14, .13, "Consistency", "NIS + confidence", YELLOW, YELLOW_D, 6.8, 5.5)
    node(ax, .37, .39, .22, .14, "Effective covariance", "visual influence decreases as disagreement grows", RED, RED_D, 6.8, 5.3)
    node(ax, .37, .18, .22, .11, "Kalman gain and update", r"$K_t \rightarrow x_t,P_t$", YELLOW, YELLOW_D, 6.8, 5.5)
    arr(ax, (.255, .69), (.33, .705), "prediction", PURPLE_D)
    arr(ax, (.255, .36), (.33, .68), "measurement", BLUE_D, -.18, label_offset=.004)
    arr(ax, (.46, .705), (.52, .705)); arr(ax, (.59, .64), (.50, .53))
    arr(ax, (.48, .39), (.48, .29))

    node(ax, .735, .61, .22, .15, "Route constraints", "progress · cross-track · step", GREEN, GREEN_D)
    node(ax, .735, .36, .22, .13, "Final route state", r"$[s,e,v_s,v_e]_t$", GREEN, GREEN_D)
    node(ax, .735, .17, .22, .09, "Causal feedback", "posterior → next frame", PURPLE, PURPLE_D)
    arr(ax, (.59, .235), (.735, .685)); arr(ax, (.845, .61), (.845, .49)); arr(ax, (.845, .36), (.845, .26))
    arr(ax, (.735, .215), (.26, .17), "next-frame posterior", PURPLE_D, -.08, .75, -.018)
    save(fig, "fig09_uncertainty_aware_kalman_fusion")


if __name__ == "__main__":
    figure6_pipeline()
    figure7_forward_selection()
    figure8_gru_motion()
    figure9_kalman()
    print(f"Wrote four publication figures to {OUT}")
