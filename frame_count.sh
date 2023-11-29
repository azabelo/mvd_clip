
#!/bin/bash

# Check if FFprobe is installed
if ! command -v ffprobe > /dev/null; then
  echo "FFprobe is not installed. Please install FFmpeg/FFprobe."
  exit 1
fi

# Check if a CSV file is provided as an argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <input_csv_file>"
  exit 1
fi

input_csv="$1"

output_csv="${input_csv%.csv}_with_frames.csv"

# Check if the input CSV file exists
if [ ! -f "$input_csv" ]; then
  echo "Input CSV file does not exist."
  exit 1
fi

# Create the output CSV file
echo "Path Frames" > "$output_csv"

# Process each line in the input CSV file
while IFS=, read -r video_path; do
  # Use ffprobe to get the number of frames
  num_frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 "$video_path" 2>/dev/null)

  # Append the video path and number of frames to the output CSV
  echo "$video_path $num_frames" >> "$output_csv"
done < "$input_csv"

echo "Frames count added to $output_csv"
