import pandas as pd
import csv

def print_and_save(models, avg_results, std_results, info, file_name):
    # ==========================================
    # Wyświetlenie wyników w konsoli
    # ==========================================

    metrics = list(avg_results[0].keys())

    print("\n" + info)
    print("\n" + "=" * 190)

    header = f"{'Model':<25} | " + " | ".join(
        [f"{m.upper():<15}" for m in metrics]
    )
    print(header)

    print("-" * 190)

    for i, (avg, std) in enumerate(zip(avg_results, std_results)):
        metric_values = " | ".join(
            [f"{avg[m]:.3f} ({std[m]:.3f})".ljust(15) for m in metrics]
        )

        print(f"{models[i]:<25} | {metric_values}")

    print("=" * 190)

    # ==========================================
    # Zapis wyników do results_main.csv
    # ==========================================

    rows = []

    for i, (avg, std) in enumerate(zip(avg_results, std_results)):

        row = {"Model": models[i]}

        for metric in avg.keys():
            row[f"{metric.upper()}_avg"] = f"{avg[metric]:.3f}"
            row[f"{metric.upper()}_std"] = f"{std[metric]:.3f}"

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        f"Output_files/{file_name}",
        index=False
    )

    print("\nResults saved to:")
    print(f"Output_files/{file_name}")


def print_ablation_test(ablation_results, csv_path=None):
    print("\n" + "=" * 110)
    print(
        f"{'Ablation scenario':<35} | "
        f"{'P_MAE (std)':<20} | "
        f"{'T_MAE (std)':<20} | "
        f"{'P_R2 (std)':<20}"
    )
    print("-" * 110)

    csv_rows = []

    for name, res in ablation_results.items():
        p_mae = f"{res[0]['pressure_mae']:.3f} ({res[1]['pressure_mae']:.3f})"
        t_mae = f"{res[0]['dT_mae']:.3f} ({res[1]['dT_mae']:.3f})"
        p_r2 = f"{res[0]['pressure_r2']:.3f} ({res[1]['pressure_r2']:.3f})"

        print(
            f"{name:<35} | "
            f"{p_mae:<20} | "
            f"{t_mae:<20} | "
            f"{p_r2:<20}"
        )

        csv_rows.append({
            "scenario": name,
            "pressure_mae": f"{res[0]['pressure_mae']:.3f} ({res[1]['pressure_mae']:.3f})",
            "dT_mae": f"{res[0]['dT_mae']:.3f} ({res[1]['dT_mae']:.3f})",
            "pressure_r2": f"{res[0]['pressure_r2']:.3f} ({res[1]['pressure_r2']:.3f})",
        })

    print("=" * 110)

    # zapis CSV
    if csv_path is not None:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"\nWyniki zapisano do CSV: {csv_path}")