"""Generate DQN architecture diagram and article EE distribution."""
import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "results/figures"

# ── DQN Architecture Diagram ─────────────────────────────────
print("dqn_architecture.png")
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(-0.5, 11)
ax.set_ylim(-1.5, 5)
ax.axis('off')

layers = [
    (0.5, 'Entrée\n$s_t \\in \\mathbb{R}^6$', 6, '#60A5FA'),
    (2.5, 'FC + ReLU\n128', 128, '#34D399'),
    (4.5, 'FC + ReLU\n256', 256, '#A78BFA'),
    (6.5, 'FC + ReLU\n128', 128, '#34D399'),
    (8.5, 'Sortie\n$Q(s,a) \\in \\mathbb{R}^{32}$', 32, '#FB923C'),
]

for x, label, neurons, color in layers:
    box_h = min(neurons / 60, 3.5) 
    box_h = max(box_h, 1.2)
    y_lo = 2 - box_h/2
    rect = FancyBboxPatch((x-0.6, y_lo), 1.2, box_h,
                          boxstyle="round,pad=0.1", facecolor=color,
                          edgecolor='gray', linewidth=1.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x, 2, label, ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows
for i in range(len(layers)-1):
    x1 = layers[i][0] + 0.6
    x2 = layers[i+1][0] - 0.6
    ax.annotate('', xy=(x2, 2), xytext=(x1, 2),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

# State components annotation
state_labels = ['$g_{kk}$', '$I_k$', '$p_k$', '$\\gamma_k$', '$R_k$', '$\\eta_{EE}$']
for i, sl in enumerate(state_labels):
    ax.text(-0.3, 3.5 - i*0.55, sl, fontsize=9, ha='right', color='#1E40AF')
    ax.annotate('', xy=(0, 2 + (2-i)*0.3), xytext=(-0.15, 3.5 - i*0.55),
                arrowprops=dict(arrowstyle='->', lw=0.8, color='#60A5FA', alpha=0.5))

# Action label
ax.text(10.2, 2, '$a_t = \\arg\\max_a Q(s_t, a)$', fontsize=10, ha='left',
        bbox=dict(boxstyle='round', facecolor='#FEF3C7', edgecolor='#F59E0B'))
ax.annotate('', xy=(10.1, 2), xytext=(9.1, 2),
            arrowprops=dict(arrowstyle='->', lw=2, color='#F59E0B'))

ax.set_title("Architecture du réseau DQN — Couches entièrement connectées", fontsize=13, pad=15)
fig.tight_layout()
fig.savefig(f"{OUT}/dqn_architecture.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ── Article EE Distribution ──────────────────────────────────
print("article_ee_distribution.png")
with open("results/article_algorithm_results.json") as f:
    article = json.load(f)

ee_vals = article['per_trial_ee_mbit_j']
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ee_vals, bins=20, color='#3B82F6', alpha=0.7, edgecolor='white')
ax.axvline(np.mean(ee_vals), color='red', linestyle='--', linewidth=2,
           label=f'Moyenne = {np.mean(ee_vals):.1f} Mbit/J')
ax.axvline(np.median(ee_vals), color='green', linestyle=':', linewidth=2,
           label=f'Médiane = {np.median(ee_vals):.1f} Mbit/J')
ax.set_xlabel('Efficacité Énergétique (Mbit/J)')
ax.set_ylabel('Fréquence')
ax.set_title("Distribution de l'EE — Algorithme Article (BRD+Dinkelbach)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/article_ee_distribution.png", dpi=300)
plt.close(fig)

print("✓ Done")
