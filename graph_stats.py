import sys
import re
import pandas as pd
import matplotlib.pyplot as plt

# Replace with the path to your file

num_args = len(sys.argv)
if num_args < 2:
    print("Profide the profile stats text file as the first command arg")
    exit()

file_path = sys.argv[1]

print(sys.argv)


# Regular expression pattern to match the data
pattern = re.compile(
    r'(?P<time>\d+\.\d+)\s+sec\s+:\s+'
    r'\[For prev \d+ ticks\]: Average: (?P<avg>\d+\.\d+)\s+msec - Min: (?P<min>\d+\.\d+)\s+msec - Max: (?P<max>\d+\.\d+)\s+msec\s+///\s+'
    r'\[For prev \d+ ticks\]: Low 10%: (?P<low10>\d+\.\d+)\s+msec - Low 1%: (?P<low1>\d+\.\d+)\s+msec - Low 0.1%: (?P<low01>\d+\.\d+)\s+msec'
)

# Load and parse the file
data = {
    'time': [], 'average': [], 'min': [], 'max': [],
    'low_10': [], 'low_1': [], 'low_0_1': []
}

with open(file_path, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            data['time'].append(float(match.group('time')))
            data['average'].append(float(match.group('avg')))
            data['min'].append(float(match.group('min')))
            data['max'].append(float(match.group('max')))
            data['low_10'].append(float(match.group('low10')))
            data['low_1'].append(float(match.group('low1')))
            data['low_0_1'].append(float(match.group('low01')))

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['average'], label='Average', marker='o')
plt.plot(df['time'], df['min'], label='Min', linestyle='--')
plt.plot(df['time'], df['max'], label='Max', linestyle='--')
plt.plot(df['time'], df['low_10'], label='Low 10%', linestyle=':')
plt.plot(df['time'], df['low_1'], label='Low 1%', linestyle=':')
plt.plot(df['time'], df['low_0_1'], label='Low 0.1%', linestyle=':')

plt.xlabel('Time (sec)')
plt.ylabel('Latency (msec)')
plt.title('Latency Profiling Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
