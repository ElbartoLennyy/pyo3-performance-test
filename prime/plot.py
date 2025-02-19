import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_benchmarks(mac_csv="benchmark_results.csv", win_csv="benchmark_results_win.csv"):
    # Read both CSV files into DataFrames
    df_mac = pd.read_csv(mac_csv)
    df_win = pd.read_csv(win_csv)
    
    # Add platform column to distinguish between Mac and Windows
    df_mac['Platform'] = 'M1 Mac'
    df_win['Platform'] = 'Windows'
    
    # Combine the dataframes
    df = pd.concat([df_mac, df_win], ignore_index=True)

    names = ['getprimes', 'lastprime', 'sumprimes', 'array_count', 'convolve', 'return', 'estimate_pi']

    # Set up a color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, 10))


    # Convert N to numeric if not already
    df['N'] = pd.to_numeric(df['N'], errors='coerce')

    # Create separate plots for each function type
    for name in names:
        # Filter for rows that contain the current name
        prime_df = df[df['Type'].str.contains(name, case=False)]
        print(f"Creating plot for {name} with data:")
        print(prime_df)
       
        # Only create plot if we have data
        if not prime_df.empty:
            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Get unique implementation types
            implementations = prime_df['Type'].unique()
            
            # Create a color dictionary to maintain consistent colors
            color_dict = {impl: color for impl, color in zip(implementations, colors)}
            
            # Plot Mac data (left subplot)
            for t in implementations:
                mac_data = prime_df[(prime_df['Type'] == t) & (prime_df['Platform'] == 'M1 Mac')].sort_values(by='N')
                if not mac_data.empty:
                    ax1.plot(mac_data['N'], mac_data['Average_Time'], 
                            color=color_dict[t],
                            marker='o', 
                            label=t)
            
            ax1.set_xlabel('N')
            ax1.set_ylabel('Average Time (s)')
            ax1.set_title(f'M1 Mac: {name} Variants')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.grid(True)
            ax1.legend()

            # Plot Windows data (right subplot)
            for t in implementations:
                win_data = prime_df[(prime_df['Type'] == t) & (prime_df['Platform'] == 'Windows')].sort_values(by='N')
                if not win_data.empty:
                    ax2.plot(win_data['N'], win_data['Average_Time'], 
                            color=color_dict[t],
                            marker='s', 
                            label=t)
            
            ax2.set_xlabel('N')
            ax2.set_ylabel('Average Time (s)')
            ax2.set_title(f'Windows: {name} Variants')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.grid(True)
            ax2.legend()

            # Adjust layout and save
            plt.suptitle(f'Performance Comparison: {name} Variants', y=1.02, fontsize=14)
            plt.tight_layout()
            plt.savefig(f'plots/{name}_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Plot saved as {name}_performance_comparison.png")
        else:
            print(f"No data found for {name}")

if __name__ == "__main__":
    plot_benchmarks()
