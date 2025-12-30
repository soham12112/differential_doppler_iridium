"""
Comprehensive Data Statistics Analysis
=======================================
Analyzes collected Iridium data for:
- Noise statistics and SINR
- Sample rate and data collection metrics  
- Signal quality per satellite
- Frame statistics
- Measurement quality
"""

import pickle
import numpy as np
import sys
import os

# Add parent directory to path if needed
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.setup import *

# Define NavDataArrayIndices locally to avoid circular imports
class NavDataArrayIndices:
    rel_time = 0
    t = rel_time
    freq = 1
    f = freq
    base_freq = 2
    fb = base_freq
    sat_id = 3
    id = sat_id
    x = 4
    y = 5
    z = 6
    vx = 7
    vy = 8
    vz = 9

IDX = NavDataArrayIndices()


class ComprehensiveDataStats:
    """Complete statistics about collected data"""
    
    def __init__(self, exp_name=None):
        self.exp_name = exp_name or EXP_NAME
        self.data_path = WORKING_DIR + f"/{self.exp_name}/"
        self.frame_file = self.data_path + FRAME_FILE
        self.nav_data_file = self.data_path + SAVED_DATA_FILE
        
    def analyze_all(self):
        """Run complete analysis and print results"""
        print("="*80)
        print(f"COMPREHENSIVE DATA ANALYSIS: {self.exp_name}")
        print("="*80)
        
        # Load data
        print("\n[1/5] Loading data...")
        nav_data = self._load_nav_data()
        frames = self._load_frames()
        
        # Basic stats
        print("\n[2/5] Analyzing basic statistics...")
        self._print_basic_stats(nav_data)
        
        # Sample rate and timing
        print("\n[3/5] Analyzing timing and sample rates...")
        self._print_timing_stats(nav_data, frames)
        
        # Noise and signal quality
        print("\n[4/5] Analyzing noise and signal quality...")
        self._print_noise_stats(frames)
        
        # Per-satellite statistics
        print("\n[5/5] Analyzing per-satellite statistics...")
        self._print_per_satellite_stats(nav_data)
        
        # Frame rate analysis
        print("\n" + "="*80)
        print("FRAME RATE ANALYSIS")
        print("="*80)
        self._print_frame_rate_stats(nav_data)
        
        # Doppler statistics
        print("\n" + "="*80)
        print("DOPPLER SHIFT STATISTICS")
        print("="*80)
        self._print_doppler_stats(nav_data)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        return nav_data, frames
    
    def _load_nav_data(self):
        """Load processed navigation data"""
        with open(self.nav_data_file, "rb") as file:
            return pickle.load(file)
    
    def _load_frames(self):
        """Load raw frames"""
        with open(self.frame_file, "r") as file:
            return file.readlines()
    
    def _print_basic_stats(self, nav_data):
        """Print basic dataset statistics"""
        print(f"\n{'Experiment Name:':<30} {self.exp_name}")
        
        # Try to load start time
        try:
            with open(self.data_path + "start_time.txt", "r") as file:
                start_time = file.readlines()[0].strip()
            print(f"{'Start Time:':<30} {start_time}")
        except:
            print(f"{'Start Time:':<30} Not found")
        
        # Data duration and size
        duration = nav_data[-1, IDX.t] - nav_data[0, IDX.t]
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        print(f"{'Total Measurements:':<30} {len(nav_data):,}")
        print(f"{'Duration:':<30} {hours:02d}h {minutes:02d}m {seconds:02d}s ({duration:.1f} seconds)")
        
        # Satellites
        unique_sats = np.unique(nav_data[:, IDX.sat_id])
        print(f"{'Number of Satellites:':<30} {len(unique_sats)}")
        print(f"{'Satellite IDs:':<30} {', '.join(str(int(s)) for s in unique_sats)}")
    
    def _print_timing_stats(self, nav_data, frames):
        """Print timing and sample rate statistics"""
        
        # Hardware sample rate (from config)
        print(f"\n{'Hardware Configuration:'}")
        print(f"  {'SDR Sample Rate:':<28} 5,000,000 samples/sec (5 MHz)")
        print(f"  {'Center Frequency:':<28} 1,626.2708 MHz")
        print(f"  {'SDR Type:':<28} PlutoSDR (ADALM-PLUTO)")
        
        # Effective frame rate
        duration = nav_data[-1, IDX.t] - nav_data[0, IDX.t]
        frame_rate = len(nav_data) / duration
        
        print(f"\n{'Effective Frame Rate:'}")
        print(f"  {'Frames per second:':<28} {frame_rate:.2f} Hz")
        print(f"  {'Average time between frames:':<28} {1000/frame_rate:.1f} ms")
        
        # Time gaps analysis
        time_diffs = np.diff(nav_data[:, IDX.t])
        
        print(f"\n{'Frame Timing Statistics:'}")
        print(f"  {'Min time gap:':<28} {np.min(time_diffs):.3f} s")
        print(f"  {'Mean time gap:':<28} {np.mean(time_diffs):.3f} s")
        print(f"  {'Max time gap:':<28} {np.max(time_diffs):.3f} s")
        print(f"  {'Std dev of time gaps:':<28} {np.std(time_diffs):.3f} s")
        
        # Find large gaps
        large_gaps = time_diffs > 60
        if np.any(large_gaps):
            print(f"  {'Large gaps (>60s):':<28} {np.sum(large_gaps)} gaps found")
    
    def _print_noise_stats(self, frames):
        """Print noise and signal quality statistics"""
        
        # Parse frame data
        ira_frames = [f for f in frames if f.startswith("IRA")]
        ibc_frames = [f for f in frames if f.startswith("IBC")]
        
        print(f"\n{'Frame Counts:'}")
        print(f"  {'Total frames:':<28} {len(frames):,}")
        print(f"  {'IRA frames (Ring Alert):':<28} {len(ira_frames):,}")
        print(f"  {'IBC frames (Broadcast):':<28} {len(ibc_frames):,}")
        print(f"  {'Other frames:':<28} {len(frames) - len(ira_frames) - len(ibc_frames):,}")
        
        if len(ira_frames) > 0:
            # Extract noise and confidence from IRA frames
            # Format: IRA: sat=XXX ... confidence% noise1|noise2|noise3|noise4
            confidences = []
            noise_levels = []
            signal_levels = []
            
            for frame in ira_frames:
                try:
                    parts = frame.split()
                    # Find confidence (has % symbol)
                    conf_idx = [i for i, p in enumerate(parts) if '%' in p]
                    if conf_idx:
                        conf = int(parts[conf_idx[0]].replace('%', ''))
                        confidences.append(conf)
                        
                        # Noise is next field (contains | symbols)
                        if conf_idx[0] + 1 < len(parts):
                            noise_str = parts[conf_idx[0] + 1]
                            if '|' in noise_str:
                                noise_vals = [float(n) for n in noise_str.split('|')]
                                noise_levels.append(noise_vals)
                                
                                # Estimate signal level from confidence
                                # Confidence is typically related to SNR
                                # Rough approximation: signal = noise * (1 + confidence/100)
                                signal_est = np.mean(noise_vals) * (1 + conf/100)
                                signal_levels.append(signal_est)
                except:
                    continue
            
            if len(confidences) > 0:
                print(f"\n{'Signal Quality Metrics (from IRA frames):'}")
                print(f"  {'Mean confidence:':<28} {np.mean(confidences):.1f}%")
                print(f"  {'Min confidence:':<28} {np.min(confidences):.1f}%")
                print(f"  {'Max confidence:':<28} {np.max(confidences):.1f}%")
                print(f"  {'Std dev confidence:':<28} {np.std(confidences):.1f}%")
                
                if len(noise_levels) > 0:
                    noise_array = np.array(noise_levels)
                    mean_noise = np.mean(noise_array, axis=0)
                    overall_noise = np.mean(noise_array)
                    
                    print(f"\n{'Noise Statistics:'}")
                    print(f"  {'Overall mean noise:':<28} {overall_noise:.2f}")
                    print(f"  {'Per-channel noise:':<28} {' | '.join(f'{n:.2f}' for n in mean_noise)}")
                    print(f"  {'Min noise:':<28} {np.min(noise_array):.2f}")
                    print(f"  {'Max noise:':<28} {np.max(noise_array):.2f}")
                    print(f"  {'Noise std dev:':<28} {np.std(noise_array):.2f}")
                    
                    # Calculate approximate SINR
                    if len(signal_levels) > 0:
                        signal_array = np.array(signal_levels)
                        sinr_db = 10 * np.log10(signal_array / overall_noise)
                        
                        print(f"\n{'Estimated SINR (Signal-to-Interference+Noise Ratio):'}")
                        print(f"  {'Mean SINR:':<28} {np.mean(sinr_db):.2f} dB")
                        print(f"  {'Min SINR:':<28} {np.min(sinr_db):.2f} dB")
                        print(f"  {'Max SINR:':<28} {np.max(sinr_db):.2f} dB")
                        print(f"  {'Std dev SINR:':<28} {np.std(sinr_db):.2f} dB")
                        
                        # Quality assessment
                        print(f"\n{'Signal Quality Assessment:'}")
                        excellent = np.sum(sinr_db > 20)
                        good = np.sum((sinr_db > 10) & (sinr_db <= 20))
                        fair = np.sum((sinr_db > 5) & (sinr_db <= 10))
                        poor = np.sum(sinr_db <= 5)
                        
                        total = len(sinr_db)
                        print(f"  {'Excellent (>20 dB):':<28} {excellent:5d} ({100*excellent/total:5.1f}%)")
                        print(f"  {'Good (10-20 dB):':<28} {good:5d} ({100*good/total:5.1f}%)")
                        print(f"  {'Fair (5-10 dB):':<28} {fair:5d} ({100*fair/total:5.1f}%)")
                        print(f"  {'Poor (<5 dB):':<28} {poor:5d} ({100*poor/total:5.1f}%)")
        else:
            print("\n  No IRA frames found for noise analysis")
    
    def _print_per_satellite_stats(self, nav_data):
        """Print statistics per satellite"""
        
        unique_sats = np.unique(nav_data[:, IDX.sat_id])
        
        print(f"\n{'Satellite':<12} {'Frames':<10} {'Duration':<12} {'Frame Rate':<12} {'Doppler Range':<20}")
        print("-" * 80)
        
        sat_stats = []
        for sat_id in unique_sats:
            mask = nav_data[:, IDX.sat_id] == sat_id
            sat_data = nav_data[mask]
            
            n_frames = len(sat_data)
            duration = sat_data[-1, IDX.t] - sat_data[0, IDX.t]
            frame_rate = n_frames / duration if duration > 0 else 0
            
            doppler = sat_data[:, IDX.f] - sat_data[:, IDX.fb]
            dopp_range = np.max(doppler) - np.min(doppler)
            
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            dur_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            print(f"{int(sat_id):<12} {n_frames:<10} {dur_str:<12} {frame_rate:>8.2f} Hz  {dopp_range/1000:>10.2f} kHz")
            
            sat_stats.append({
                'id': int(sat_id),
                'frames': n_frames,
                'duration': duration,
                'frame_rate': frame_rate,
                'doppler_range': dopp_range
            })
        
        # Summary statistics
        total_frames = sum(s['frames'] for s in sat_stats)
        mean_frames = np.mean([s['frames'] for s in sat_stats])
        mean_duration = np.mean([s['duration'] for s in sat_stats])
        
        print("-" * 80)
        print(f"{'SUMMARY':<12} {total_frames:<10} {mean_duration:>8.1f}s avg")
    
    def _print_frame_rate_stats(self, nav_data):
        """Detailed frame rate analysis"""
        
        # Overall frame rate
        duration = nav_data[-1, IDX.t] - nav_data[0, IDX.t]
        overall_rate = len(nav_data) / duration
        
        print(f"\n{'Overall Frame Rate:':<30} {overall_rate:.3f} frames/second")
        
        # Frame rate in sliding windows
        window_sizes = [60, 300, 600]  # 1 min, 5 min, 10 min
        
        for window in window_sizes:
            rates = []
            t_start = nav_data[0, IDX.t]
            t_end = nav_data[-1, IDX.t]
            
            t = t_start
            while t + window <= t_end:
                mask = (nav_data[:, IDX.t] >= t) & (nav_data[:, IDX.t] < t + window)
                n_frames = np.sum(mask)
                rate = n_frames / window
                rates.append(rate)
                t += window / 2  # 50% overlap
            
            if len(rates) > 0:
                print(f"\n{f'Frame Rate ({window}s windows):':<30}")
                print(f"  {'Mean:':<28} {np.mean(rates):.3f} frames/s")
                print(f"  {'Min:':<28} {np.min(rates):.3f} frames/s")
                print(f"  {'Max:':<28} {np.max(rates):.3f} frames/s")
                print(f"  {'Std dev:':<28} {np.std(rates):.3f} frames/s")
    
    def _print_doppler_stats(self, nav_data):
        """Doppler shift statistics"""
        
        doppler = nav_data[:, IDX.f] - nav_data[:, IDX.fb]
        
        print(f"\n{'Doppler Shift Range:':<30} {np.min(doppler)/1000:.2f} to {np.max(doppler)/1000:.2f} kHz")
        print(f"{'Mean Doppler:':<30} {np.mean(doppler)/1000:.2f} kHz")
        print(f"{'Std Dev Doppler:':<30} {np.std(doppler)/1000:.2f} kHz")
        
        # Doppler rate (derivative) - calculate per satellite
        doppler_rates = []
        unique_sats = np.unique(nav_data[:, IDX.sat_id])
        
        for sat_id in unique_sats:
            mask = nav_data[:, IDX.sat_id] == sat_id
            sat_data = nav_data[mask]
            
            if len(sat_data) > 2:
                sat_doppler = sat_data[:, IDX.f] - sat_data[:, IDX.fb]
                sat_time = sat_data[:, IDX.t]
                
                # Calculate rate of change
                d_dopp = np.diff(sat_doppler)
                d_time = np.diff(sat_time)
                
                # Only consider reasonable time gaps (< 60 seconds)
                valid_mask = d_time < 60
                rates = d_dopp[valid_mask] / d_time[valid_mask]
                doppler_rates.extend(rates[np.abs(rates) < 1000])  # Filter outliers
        
        if len(doppler_rates) > 0:
            print(f"\n{'Doppler Rate Statistics:'}")
            print(f"  {'Mean rate:':<28} {np.mean(doppler_rates):.2f} Hz/s")
            print(f"  {'Std dev rate:':<28} {np.std(doppler_rates):.2f} Hz/s")
            print(f"  {'Max rate:':<28} {np.max(np.abs(doppler_rates)):.2f} Hz/s")


def main():
    """Run comprehensive data statistics analysis"""
    analyzer = ComprehensiveDataStats()
    nav_data, frames = analyzer.analyze_all()
    
    # Optional: Create plots
    print("\nWould you like to generate plots? (requires matplotlib)")
    print("You can manually create plots using the returned nav_data and frames")
    
    return nav_data, frames


if __name__ == "__main__":
    main()

