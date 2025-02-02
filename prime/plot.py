import pandas as pd
import matplotlib.pyplot as plt

def plot_benchmarks(csv_file="benchmark_results.csv"):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    names = ['getprimes', 'lastprime', 'sumprimes', 'array_count','convolve','return','estimate_pi']

    # We assume the CSV has the following columns:
    # ['Function_Name', 'Type', 'N', 'Average_Time', 'Output_Length']

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
            plt.figure(figsize=(10, 6))
            for t in prime_df['Type'].unique():
                subset = prime_df[prime_df['Type'] == t].sort_values(by='N')
                plt.plot(subset['N'], subset['Average_Time'], marker='o', label=t)

            plt.xlabel('N')
            plt.ylabel('Average Time (s)')
            plt.title(f'Performance Comparison: {name} Variants')
            plt.xscale('log')  # Set the x-axis to log scale
            plt.yscale("log")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'plots/{name}_performance.png', dpi=300)
            plt.close()

            print(f"Plot saved as {name}_performance.png")
        else:
            print(f"No data found for {name}")

if __name__ == "__main__":
    plot_benchmarks()
