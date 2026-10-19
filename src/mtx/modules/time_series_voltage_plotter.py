import pandas as pd
import matplotlib.pyplot as plt

def plot_voltage_pair(df_volt, meter_a, meter_b, day):
    """
    Plot three voltage graphs:
    1. Combined (both meters on same plot, no offset)
    2. Individual Meter A
    3. Individual Meter B
    """

    # --- Data prep ---
    df_bis = df_volt.copy()
    df_bis["Timestamp"] = pd.to_datetime(df_bis["Timestamp"])
    df_bis["Date"] = pd.to_datetime(df_bis["DayRecorded"]).dt.strftime('%Y-%m-%d')
    df_day = df_bis[df_bis["Date"] == day].reset_index(drop=True)

    df_a = df_day[df_day["Meter_Number"] == str(meter_a)].sort_values("Timestamp")
    df_b = df_day[df_day["Meter_Number"] == str(meter_b)].sort_values("Timestamp")

    # --- Create figure with 3 subplots ---
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.suptitle(f"Voltage Comparison — {day}", fontsize=14, fontweight='bold')

    # --- (1) Combined graph ---
    axes[0].plot(df_a["Timestamp"], df_a["VoltageReading"], marker="o", color="tab:blue", label=f"Meter {meter_a}")
    axes[0].plot(df_b["Timestamp"], df_b["VoltageReading"], marker="s", color="tab:orange", label=f"Meter {meter_b}")
    axes[0].set_title("Combined Voltage Readings")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].legend()
    axes[0].grid(True)

    # --- (2) Individual meter A ---
    axes[1].plot(df_a["Timestamp"], df_a["VoltageReading"], marker="o", color="tab:blue")
    axes[1].set_title(f"Meter {meter_a}")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].grid(True)

    # --- (3) Individual meter B ---
    axes[2].plot(df_b["Timestamp"], df_b["VoltageReading"], marker="s", color="tab:orange")
    axes[2].set_title(f"Meter {meter_b}")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Voltage (V)")
    axes[2].grid(True)

    # --- Layout ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for the main title
    plt.show()