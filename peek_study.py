"""Quick read-only peek at the Optuna study progress."""
import sqlite3

conn = sqlite3.connect("file:optuna_study.db?mode=ro", uri=True)
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE'")
n_complete = cur.fetchone()[0]
print(f"Completed trials: {n_complete}/100")

cur.execute("""
    SELECT t.number, tv0.value, tv1.value
    FROM trials t
    JOIN trial_values tv0 ON t.trial_id = tv0.trial_id AND tv0.objective = 0
    JOIN trial_values tv1 ON t.trial_id = tv1.trial_id AND tv1.objective = 1
    WHERE t.state = 'COMPLETE'
    ORDER BY t.number
""")
results = cur.fetchall()

f1s = [r[1] for r in results]
maps = [r[2] for r in results]

print(f"\n--- F1 Score ---")
print(f"  Best:  {max(f1s):.4f}")
print(f"  Worst: {min(f1s):.4f}")
print(f"  Mean:  {sum(f1s)/len(f1s):.4f}")

print(f"\n--- mAP50 ---")
print(f"  Best:  {max(maps):.4f}")
print(f"  Worst: {min(maps):.4f}")
print(f"  Mean:  {sum(maps)/len(maps):.4f}")

print(f"\n--- Top 10 Trials by F1 ---")
by_f1 = sorted(results, key=lambda x: x[1], reverse=True)[:10]
for trial_num, f1, map50 in by_f1:
    print(f"  Trial {trial_num:3d}: F1={f1:.4f}  mAP50={map50:.4f}")

print(f"\n--- Top 10 Trials by mAP50 ---")
by_map = sorted(results, key=lambda x: x[2], reverse=True)[:10]
for trial_num, f1, map50 in by_map:
    print(f"  Trial {trial_num:3d}: F1={f1:.4f}  mAP50={map50:.4f}")

pareto = []
for r in results:
    dominated = False
    for other in results:
        if other[1] > r[1] and other[2] > r[2]:
            dominated = True
            break
    if not dominated:
        pareto.append(r)
pareto.sort(key=lambda x: x[1], reverse=True)

print(f"\n--- Pareto Frontier ({len(pareto)} trials) ---")
for trial_num, f1, map50 in pareto:
    print(f"  Trial {trial_num:3d}: F1={f1:.4f}  mAP50={map50:.4f}")

# Get params for the best Pareto trials
print("\n--- Best Pareto Trial Details ---")
for trial_num, f1, map50 in pareto[:3]:
    cur.execute("""
        SELECT tp.param_name, tp.param_value
        FROM trials t
        JOIN trial_params tp ON t.trial_id = tp.trial_id
        WHERE t.number = ?
        ORDER BY tp.param_name
    """, (trial_num,))
    params = cur.fetchall()
    print(f"\n  Trial {trial_num} (F1={f1:.4f}, mAP50={map50:.4f}):")
    for name, value in params:
        print(f"    {name:20s} = {value}")

conn.close()
