import asyncio
import os
import json
import csv
import sys
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import argparse

# Check for required dependencies
try:
    import librosa
    import numpy as np
    from shazamio import Shazam
    import soundfile as sf
    from scipy.signal import find_peaks
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("Install with: pip install shazamio librosa soundfile scipy numpy")
    sys.exit(1)


class DJSetAnalyzer:
    def __init__(self,
                 segment_duration: int = 30,
                 overlap: int = 10,
                 min_confidence: float = 0.3,
                 rate_limit_delay: int = 3):

        self.segment_duration = segment_duration
        self.overlap = overlap
        self.min_confidence = min_confidence
        self.rate_limit_delay = rate_limit_delay
        self.shazam = Shazam()
        self.temp_dir = None

    def __enter__(self):
        """Context manager entry"""
        self.temp_dir = tempfile.mkdtemp()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temp files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"⚠️ Warning: Could not clean up temp directory: {e}")

    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        if seconds < 0:
            seconds = 0
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def detect_audio_changes(self, audio_file: str) -> List[float]:
        """
        Detect potential track boundaries using audio analysis
        Returns list of timestamps where significant changes occur
        """
        print("🎵 Analyzing audio for track boundaries...")

        try:
            # Load audio file with error handling
            y, sr = librosa.load(audio_file, sr=22050)

            if len(y) == 0:
                print("⚠️ Warning: Audio file appears to be empty")
                return []

            print(f"   📊 Audio loaded: {len(y) / sr:.1f}s at {sr}Hz")

            # Calculate spectral features for change detection
            hop_length = 512

            # Chroma features (key/harmony changes)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

            # MFCC features (timbre changes)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

            # Spectral centroid (brightness changes)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)

            # Combine features - ensure they have the same number of frames
            min_frames = min(chroma.shape[1], mfcc.shape[1], spectral_centroid.shape[1])
            features = np.vstack([
                chroma[:, :min_frames],
                mfcc[:, :min_frames],
                spectral_centroid[:, :min_frames]
            ])

            # Calculate feature difference between adjacent frames
            if features.shape[1] > 1:
                feature_diff = np.diff(features, axis=1)
                feature_change = np.sum(np.abs(feature_diff), axis=0)

                # Convert frame indices to time
                times = librosa.frame_to_time(np.arange(len(feature_change)), sr=sr, hop_length=hop_length)

                # Find peaks in feature changes (potential track boundaries)
                if len(feature_change) > 0:
                    threshold = np.percentile(feature_change, 75)  # Top 25% of changes
                    min_distance = int(sr * 30 / hop_length)  # At least 30s apart

                    peaks, _ = find_peaks(feature_change,
                                          height=threshold,
                                          distance=min_distance)

                    boundary_times = times[peaks].tolist()
                else:
                    boundary_times = []
            else:
                boundary_times = []

            print(f"🎯 Detected {len(boundary_times)} potential track boundaries")
            return boundary_times

        except Exception as e:
            print(f"⚠️ Error in audio analysis: {e}")
            return []

    def create_audio_segments(self, audio_file: str, boundary_times: List[float] = None) -> List[
        Tuple[float, float, str]]:
        """
        Create audio segments for identification
        Returns list of (start_time, end_time, temp_file_path) tuples
        """
        print("✂️ Creating audio segments...")

        try:
            # Load audio to get duration
            y, sr = librosa.load(audio_file, sr=None)
            total_duration = len(y) / sr

            print(f"   📏 Total duration: {self.format_timestamp(total_duration)}")

            segments = []

            # Create regular interval segments for complete coverage
            current_time = 0
            while current_time < total_duration:
                end_time = min(current_time + self.segment_duration, total_duration)

                # Only add segments that are at least 10 seconds long
                if end_time - current_time >= 10:
                    segments.append((current_time, end_time))

                current_time += self.segment_duration - self.overlap

            # Add boundary-based segments if available
            if boundary_times:
                for boundary in boundary_times:
                    # Create segments around each boundary
                    start_time = max(0, boundary - self.segment_duration / 2)
                    end_time = min(total_duration, boundary + self.segment_duration / 2)

                    if end_time - start_time >= 10:
                        segments.append((start_time, end_time))

            # Remove duplicates and sort
            # Use a tolerance for near-duplicate segments
            unique_segments = []
            for start, end in sorted(segments):
                is_duplicate = False
                for existing_start, existing_end in unique_segments:
                    if (abs(start - existing_start) < 5 and abs(end - existing_end) < 5):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_segments.append((start, end))

            segments = unique_segments

            # Create temporary audio files for each segment
            segment_files = []

            for i, (start_time, end_time) in enumerate(segments):
                try:
                    # Extract segment
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)

                    # Ensure we don't go beyond array bounds
                    start_sample = max(0, min(start_sample, len(y)))
                    end_sample = max(start_sample, min(end_sample, len(y)))

                    segment_audio = y[start_sample:end_sample]

                    if len(segment_audio) == 0:
                        print(f"⚠️ Warning: Empty segment at {self.format_timestamp(start_time)}")
                        continue

                    # Save to temporary file
                    temp_file = os.path.join(self.temp_dir, f"segment_{i:04d}.wav")
                    sf.write(temp_file, segment_audio, sr)

                    segment_files.append((start_time, end_time, temp_file))

                except Exception as e:
                    print(f"⚠️ Error creating segment {i}: {e}")
                    continue

            print(f"📦 Created {len(segment_files)} valid segments")
            return segment_files

        except Exception as e:
            print(f"❌ Error in segment creation: {e}")
            return []

    async def identify_segment(self, start_time: float, end_time: float, audio_file: str) -> Dict:
        """Identify a single audio segment"""
        try:
            # Verify file exists before trying to identify
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Segment file not found: {audio_file}")

            result = await self.shazam.recognize(audio_file)

            if result and isinstance(result, dict) and 'track' in result:
                track = result['track']

                # Extract track information safely
                title = track.get('title', 'Unknown Title')
                artist = track.get('subtitle', 'Unknown Artist')
                shazam_id = track.get('key')

                # Try to get additional metadata
                album = None
                year = None
                if 'sections' in track and isinstance(track['sections'], list):
                    for section in track['sections']:
                        if 'metadata' in section and isinstance(section['metadata'], list):
                            for meta in section['metadata']:
                                if isinstance(meta, dict):
                                    if meta.get('title') == 'Album':
                                        album = meta.get('text')
                                    elif meta.get('title') == 'Released':
                                        year = meta.get('text')

                return {
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_timestamp': self.format_timestamp(start_time),
                    'end_timestamp': self.format_timestamp(end_time),
                    'title': title,
                    'artist': artist,
                    'shazam_id': shazam_id,
                    'album': album,
                    'year': year,
                    'status': 'identified'
                }
            else:
                return {
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_timestamp': self.format_timestamp(start_time),
                    'end_timestamp': self.format_timestamp(end_time),
                    'status': 'no_match'
                }

        except Exception as e:
            return {
                'start_time': start_time,
                'end_time': end_time,
                'start_timestamp': self.format_timestamp(start_time),
                'end_timestamp': self.format_timestamp(end_time),
                'status': 'error',
                'error': str(e)
            }

    def merge_consecutive_tracks(self, results: List[Dict]) -> List[Dict]:
        """Merge consecutive segments that identified the same track"""
        if not results:
            return results

        merged = []
        current_track = None

        # Sort by start time
        results = sorted(results, key=lambda x: x.get('start_time', 0))

        for result in results:
            if result.get('status') != 'identified':
                # Close current track and add unidentified segment
                if current_track:
                    merged.append(current_track)
                    current_track = None
                merged.append(result)
                continue

            # Check if this is the same track as current
            if (current_track and
                    current_track.get('title') == result.get('title') and
                    current_track.get('artist') == result.get('artist') and
                    result.get('start_time', 0) - current_track.get('end_time', 0) < 60):  # Within 60 seconds

                # Extend current track
                current_track['end_time'] = result['end_time']
                current_track['end_timestamp'] = result['end_timestamp']
            else:
                # New track
                if current_track:
                    merged.append(current_track)
                current_track = result.copy()

        # Don't forget the last track
        if current_track:
            merged.append(current_track)

        return merged

    async def analyze_dj_set(self, audio_file: str, output_dir: Optional[str] = None) -> Dict:
        """Main function to analyze a DJ set"""

        # Validate input file
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Check file size (warn if very large)
        file_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
        if file_size > 500:  # > 500MB
            print(f"⚠️ Warning: Large file ({file_size:.1f}MB) - this may take a long time")

        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(audio_file) or "."

        os.makedirs(output_dir, exist_ok=True)

        print(f"🎧 Analyzing DJ set: {os.path.basename(audio_file)}")
        print(f"📁 Output directory: {output_dir}")
        print("=" * 60)

        try:
            # Step 1: Detect potential track boundaries
            boundary_times = self.detect_audio_changes(audio_file)

            # Step 2: Create segments
            segments = self.create_audio_segments(audio_file, boundary_times)

            if not segments:
                raise ValueError("No valid segments could be created from the audio file")

            # Step 3: Identify each segment
            print(f"🔍 Identifying {len(segments)} segments...")
            results = []

            for i, (start_time, end_time, temp_file) in enumerate(segments):
                print(
                    f"\n[{i + 1:3d}/{len(segments)}] {self.format_timestamp(start_time)} - {self.format_timestamp(end_time)}",
                    end=" ")

                result = await self.identify_segment(start_time, end_time, temp_file)

                if result['status'] == 'identified':
                    print(f"✅ {result['title']} - {result['artist']}")
                elif result['status'] == 'no_match':
                    print("❌ No match")
                else:
                    print(f"⚠️ Error: {result.get('error', 'Unknown')}")

                results.append(result)

                # Rate limiting (don't wait after the last segment)
                if i < len(segments) - 1:
                    await asyncio.sleep(self.rate_limit_delay)

            # Step 4: Merge consecutive identical tracks
            print(f"\n🔄 Merging consecutive tracks...")
            merged_results = self.merge_consecutive_tracks(results)

            # Step 5: Generate output files
            base_name = os.path.splitext(os.path.basename(audio_file))[0]

            # Create safe filename (remove problematic characters)
            safe_base_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_')).strip()
            if not safe_base_name:
                safe_base_name = "djset_analysis"

            output_files = {}

            # JSON output
            json_file = os.path.join(output_dir, f"{safe_base_name}_tracklist.json")
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'audio_file': os.path.basename(audio_file),
                        'analysis_date': datetime.now().isoformat(),
                        'settings': {
                            'segment_duration': self.segment_duration,
                            'overlap': self.overlap,
                            'rate_limit_delay': self.rate_limit_delay
                        },
                        'stats': {
                            'total_segments_analyzed': len(segments),
                            'tracks_identified': len([r for r in merged_results if r.get('status') == 'identified']),
                            'total_tracks_in_output': len(merged_results)
                        },
                        'tracks': merged_results
                    }, f, indent=2, ensure_ascii=False)
                output_files['json'] = json_file
            except Exception as e:
                print(f"⚠️ Could not save JSON file: {e}")

            # CSV output
            csv_file = os.path.join(output_dir, f"{safe_base_name}_tracklist.csv")
            try:
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Start Time', 'End Time', 'Duration', 'Artist', 'Title', 'Album', 'Year', 'Status',
                                     'Shazam ID'])

                    for track in merged_results:
                        duration_seconds = track.get('end_time', 0) - track.get('start_time', 0)
                        duration = self.format_timestamp(duration_seconds)
                        writer.writerow([
                            track.get('start_timestamp', ''),
                            track.get('end_timestamp', ''),
                            duration,
                            track.get('artist', ''),
                            track.get('title', ''),
                            track.get('album', ''),
                            track.get('year', ''),
                            track.get('status', ''),
                            track.get('shazam_id', '')
                        ])
                output_files['csv'] = csv_file
            except Exception as e:
                print(f"⚠️ Could not save CSV file: {e}")

            # Text output (human readable)
            txt_file = os.path.join(output_dir, f"{safe_base_name}_tracklist.txt")
            try:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(f"DJ SET TRACKLIST\n")
                    f.write(f"File: {os.path.basename(audio_file)}\n")
                    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 60 + "\n\n")

                    track_num = 1
                    for track in merged_results:
                        duration_seconds = track.get('end_time', 0) - track.get('start_time', 0)
                        duration = self.format_timestamp(duration_seconds)

                        if track.get('status') == 'identified':
                            f.write(
                                f"{track_num:02d}. [{track.get('start_timestamp', '')}] {track.get('artist', 'Unknown')} - {track.get('title', 'Unknown')} ({duration})\n")
                            track_num += 1
                        else:
                            f.write(f"    [{track.get('start_timestamp', '')}] [Unknown Track] ({duration})\n")
                output_files['txt'] = txt_file
            except Exception as e:
                print(f"⚠️ Could not save TXT file: {e}")

            # Calculate statistics
            identified_count = len([r for r in merged_results if r.get('status') == 'identified'])
            total_count = len(merged_results)
            success_rate = identified_count / total_count if total_count > 0 else 0

            # Summary
            print(f"\n" + "=" * 60)
            print(f"📊 ANALYSIS COMPLETE")
            print(f"✅ Tracks identified: {identified_count}/{total_count} ({success_rate:.1%})")
            print(f"📁 Files saved:")
            for file_type, file_path in output_files.items():
                print(f"   📄 {file_type.upper()}: {file_path}")

            return {
                'tracks': merged_results,
                'stats': {
                    'total_tracks': total_count,
                    'identified_tracks': identified_count,
                    'success_rate': success_rate,
                    'segments_analyzed': len(segments)
                },
                'files': output_files
            }

        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise


async def main():
    parser = argparse.ArgumentParser(
        description='Analyze DJ set and identify tracks with timestamps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dj_analyzer.py "djset.mp3"
  python dj_analyzer.py "djset.wav" -o "./results" -s 45 -r 5
        """
    )

    parser.add_argument('audio_file', help='Path to the DJ set audio file')
    parser.add_argument('--output-dir', '-o', help='Output directory for results (default: same as input file)')
    parser.add_argument('--segment-duration', '-s', type=int, default=30,
                        help='Duration of each segment in seconds (default: 30)')
    parser.add_argument('--overlap', type=int, default=10,
                        help='Overlap between segments in seconds (default: 10)')
    parser.add_argument('--rate-limit', '-r', type=int, default=3,
                        help='Delay between API requests in seconds (default: 3)')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file not found: {args.audio_file}")
        return 1

    if args.segment_duration < 10:
        print("❌ Error: Segment duration must be at least 10 seconds")
        return 1

    if args.overlap >= args.segment_duration:
        print("❌ Error: Overlap must be less than segment duration")
        return 1

    # Use context manager to ensure cleanup
    try:
        with DJSetAnalyzer(
                segment_duration=args.segment_duration,
                overlap=args.overlap,
                rate_limit_delay=args.rate_limit
        ) as analyzer:

            results = await analyzer.analyze_dj_set(args.audio_file, args.output_dir)
            print(f"\n🎉 Analysis completed successfully!")
            print(f"📈 Success rate: {results['stats']['success_rate']:.1%}")

            return 0

    except KeyboardInterrupt:
        print(f"\n⚠️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user")
        sys.exit(1)