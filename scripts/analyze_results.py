"""Analyze training results from the 10K episode run."""
import json
import numpy as np

with open('results/training_history.json') as f:
    h = json.load(f)

ee = h['episode_ee']
loss = h['episode_losses']
sinr = h['episode_sinr_dB']
power = h['episode_powers_dBm']
qos = h['episode_qos_violations']
eps = h['epsilon']
rewards = h['episode_rewards']

N = len(ee)
print(f'Total episodes: {N}')
print()

segments = [
    ('Ep 1-1000',     0, 1000),
    ('Ep 1000-3000',  1000, 3000),
    ('Ep 3000-5000',  3000, 5000),
    ('Ep 5000-7000',  5000, 7000),
    ('Ep 7000-9000',  7000, 9000),
    ('Ep 9000-10000', 9000, 10000),
]

header = f"{'Phase':<16} | {'EE_MbitJ':>10} | {'Loss':>9} | {'SINR_dB':>9} | {'P_dBm':>8} | {'QoS_pct':>7} | {'Reward':>8} | {'Epsilon':>7}"
print(header)
print('-' * len(header))
for name, s, e in segments:
    se = ee[s:e]; sl = loss[s:e]; ss = sinr[s:e]
    sp = power[s:e]; sq = qos[s:e]; sr = rewards[s:e]; sep = eps[s:e]
    print(f'{name:<16} | {np.mean(se):10.2f} | {np.mean(sl):9.4f} | '
          f'{np.mean(ss):9.1f} | {np.mean(sp):8.1f} | {np.mean(sq)*100:6.1f}% | '
          f'{np.mean(sr):8.2f} | {sep[-1]:.4f}')

print()
print('=== Final 500 episodes (converged policy) ===')
final = slice(-500, None)
print(f'  EE:     {np.mean(ee[final]):.2f} +/- {np.std(ee[final]):.2f} Mbit/J')
print(f'  Loss:   {np.mean(loss[final]):.4f} +/- {np.std(loss[final]):.4f}')
print(f'  SINR:   {np.mean(sinr[final]):.1f} +/- {np.std(sinr[final]):.1f} dB')
print(f'  Power:  {np.mean(power[final]):.1f} +/- {np.std(power[final]):.1f} dBm')
print(f'  QoS violations: {np.mean(qos[final])*100:.1f}%')
print(f'  Reward: {np.mean(rewards[final]):.2f}')
print(f'  Best EE ever: {max(ee):.2f} Mbit/J (ep {np.argmax(ee)+1})')
print(f'  Best EE (last 500): {max(ee[final]):.2f} Mbit/J')
print(f'  Worst EE (last 500): {min(ee[final]):.2f} Mbit/J')
print()

print('=== Loss trend (last 2000 eps, 500-ep windows) ===')
for i in range(4):
    s = -2000 + i*500
    e = s + 500
    sl = loss[s:e] if e != 0 else loss[s:]
    print(f'  Window {i+1}: loss = {np.mean(sl):.4f}')
print()

print('=== EE distribution (last 500 eps) ===')
ee_f = np.array(ee[-500:])
for pct in [10, 25, 50, 75, 90]:
    print(f'  {pct}th percentile: {np.percentile(ee_f, pct):.1f} Mbit/J')
print()

# Convergence check
print('=== Convergence diagnostics ===')
ee_last2k = np.array(ee[-2000:])
ee_first = np.mean(ee_last2k[:500])
ee_last  = np.mean(ee_last2k[-500:])
print(f'  EE trend (last 2K): {ee_first:.1f} -> {ee_last:.1f} (delta={ee_last-ee_first:+.1f})')

loss_last2k = np.array(loss[-2000:])
l_first = np.mean(loss_last2k[:500])
l_last  = np.mean(loss_last2k[-500:])
print(f'  Loss trend (last 2K): {l_first:.4f} -> {l_last:.4f} (delta={l_last-l_first:+.4f})')

qos_last2k = np.array(qos[-2000:])
q_first = np.mean(qos_last2k[:500])*100
q_last  = np.mean(qos_last2k[-500:])*100
print(f'  QoS violations (last 2K): {q_first:.1f}% -> {q_last:.1f}% (delta={q_last-q_first:+.1f}%)')
