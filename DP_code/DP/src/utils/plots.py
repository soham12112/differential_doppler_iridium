"""
    Plotting helpers to reduce clutter
"""

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

from src.navigation.calculations import latlon_distance
from src.navigation.data_processing import nav_data_to_array
from src.utils.data import load_data, get_fig_filename
from src.config.locations import LOCATIONS


class ColorPicker:
    def_list = ["b", "g", "r", "c", "m", "y", "k", "aqua", "aquamarine", "blue", "blueviolet",
                "brown", "burlywood", "cadetblue", "chartreuse", "chocolate",
                "coral", "cornflowerblue", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray",
                "darkgreen", "darkgrey", "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange", "darkorchid",
                "darkred", "darksalmon", "darkseagreen", "darkslateblue", "darkslategray", "darkslategrey",
                "darkturquoise", "darkviolet", "deeppink", "deepskyblue", "dimgray", "dimgrey", "dodgerblue",
                "firebrick", "forestgreen", "fuchsia", "gold", "goldenrod", "gray", "green", "greenyellow",
                "grey", "hotpink", "indianred", "indigo", "khaki", "lavender", "lavenderblush", "lawngreen",
                "lightblue", "lightcoral", "lightcyan", "lightgreen", "lightpink", "lightsalmon", "lightseagreen",
                "lightskyblue", "lightslategray", "lightslategrey", "lightsteelblue", "lightyellow", "lime",
                "limegreen", "magenta", "maroon", "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple",
                "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise", "mediumvioletred",
                "midnightblue", "mintcream", "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive",
                "olivedrab", "orange", "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise",
                "palevioletred", "peachpuff", "peru", "pink", "plum", "powderblue", "purple", "rebeccapurple", "red",
                "rosybrown", "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen", "sienna", "silver",
                "skyblue", "slateblue", "slategray", "slategrey", "springgreen", "steelblue", "tan", "teal",
                "thistle", "tomato", "turquoise", "violet", "yellow", "yellowgreen"]

    def __init__(self):
        self.colors_by_line = dict()
        self.idx = 0
        self.color_list = self.def_list

    def _choose_new_color(self):
        if self.idx >= len(self.color_list):
            print("Warning: recycling colors")
            self.index = 0

        color = self.color_list[self.idx]
        self.idx += 1
        return color

    def reserve_colors(self, *colors):
        for color in colors:
            self.color_list.remove(color)

    def pick_color(self, line_name):
        if line_name in self.colors_by_line:
            color = self.colors_by_line[line_name]
        else:
            color = self._choose_new_color()
            print(f"Color for {line_name}: {color}")
            self.colors_by_line[line_name] = color

        return color


def plot_results_of_iterative_position_finding(data: str | list, r=None, show=False, map_covers_tracks=False):
    if isinstance(data, str):
        results = load_data(data)
    else:
        results = data

    plt.style.use("Plots/ctuthesis.mplstyle")

    final_lat, final_lon, final_alt = results[-1][2], results[-1][3], results[-1][4]
    try:
        final_off, final_dft = results[-1][5], results[-1][6]
    except IndexError:
        final_off, final_dft = None, None
    home_lon, home_lat, home_alt = LOCATIONS["HOME"][0], LOCATIONS["HOME"][1], LOCATIONS["HOME"][2]
    pos_error = latlon_distance(home_lat, final_lat, home_lon, final_lon, home_alt, final_alt)
    res_arr = np.array(results)

    min_lat = 51
    min_lon = 12
    max_lat = 54
    max_lon = 18

    if map_covers_tracks and r is not None:
        sat_track = r.earth_location.geodetic
        min_lon = min(sat_track.lon.min().value, min_lon)
        max_lat = max(sat_track.lat.max().value, max_lat)
        max_lon = max(sat_track.lon.max().value, max_lon)
        min_lat = min(sat_track.lat.min().value, min_lat)

    plt.figure()
    plt.title(f"Pos. error: {pos_error:.1f} m")
    # plt.figtext(.15, .05, f"Lat.: {final_lat:.2f}°, Lon.: {final_lon:.2f}°, Alt.: {final_alt:.0f} m" +
    #             f", Offset: {final_off / 1e3:.2f} kHz, Drift: {final_dft * 1e3:.0f} mHz/s" if final_off is not None else "")

    m = Basemap(llcrnrlon=min_lon - 2, llcrnrlat=min_lat - 2, urcrnrlon=max_lon + 2, urcrnrlat=max_lat + 2,
                rsphere=(6378137.00, 6356752.3142), resolution='h', projection='merc', )

    if r is not None:
        sat_track = r.earth_location.geodetic
        m.plot(sat_track.lon, sat_track.lat, ".", color="red", ms=1.5, latlon=True, label="Satellite track")
    m.plot(res_arr[:, 3], res_arr[:, 2], marker=".", color="green", latlon=True, label="Algorithm path")
    m.plot(final_lon, final_lat, "o", color="orange", latlon=True, label="Estimated position")
    m.plot(home_lon, home_lat, "x", color="blue", latlon=True, label="Actual position")
    m.drawcoastlines()
    m.fillcontinents()
    m.drawcountries()
    m.drawrivers(color="blue")
    m.drawparallels(np.arange(40, 65, 2), labels=[0, 1, 0, 0])
    m.drawmeridians(np.arange(0, 30, 2), labels=[0, 0, 0, 1])
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(get_fig_filename("fig"))

    if show:
        plt.show()


def plot_analyzed_curve(curve, dopp_start, dopp_end, curve_duration, curve_density, largest_gap, variance, ok=None):
    if ok is True:
        color = "green"
    elif ok is False:
        color = "red"
    else:
        color = "blue"

    curve_array = nav_data_to_array(curve)
    plt.figure()
    plt.title(f"T:{(dopp_start > 0 > dopp_end or dopp_start < 0 < dopp_end)}, "
              f"L:{curve_duration:.1f}, "
              f"D:{curve_density:.3f} \n"
              f"G{largest_gap:.2f}, "
              f"V{variance:.3f}\n")
    plt.plot(curve_array[:, 0], curve_array[:, 1] - curve_array[:, 2], ".", color=color)
    plt.savefig(get_fig_filename("analyzed_curve"))
    plt.close()


def plot_measured_vs_trial_curve(measured_curve, trial_curve, lat, lon, alt, off):
    plt.style.use("Plots/ctuthesis.mplstyle")
    plt.figure()
    plt.xlabel("Unix time [s]")
    plt.ylabel("Doppler shift [Hz]")
    # plt.figtext(.15, .12, f"Lat.: {lat:.2f}°, Lon.: {lon:.2f}°, Alt.: {alt:.0f} m, Offset: {off / 1e3:.0f} kHz")
    plt.scatter(measured_curve[:, 0], measured_curve[:, 1], marker=".", label="Measured curve")
    plt.scatter(trial_curve[:, 0], trial_curve[:, 1], marker=".", label="Trial curve")
    plt.legend()


def plot_measured_vs_predicted_with_true_location(curve_array, lat_est, lon_est, alt_est, off_est, dft_est, 
                                                   lat_true, lon_true, alt_true, satellites=None, show=False):
    """
    Plot measured vs predicted Doppler using both estimated and true locations
    
    :param curve_array: array with columns [t, f, fb, sat_id, x, y, z, vx, vy, vz, ...]
    :param lat_est: estimated latitude
    :param lon_est: estimated longitude  
    :param alt_est: estimated altitude
    :param off_est: estimated offset
    :param dft_est: estimated drift
    :param lat_true: true latitude
    :param lon_true: true longitude
    :param alt_true: true altitude
    :param satellites: dictionary of satellite objects (for names)
    :param show: whether to show the plot
    """
    from astropy.coordinates import EarthLocation
    from src.navigation.data_processing import NavDataArrayIndices as IDX
    from src.navigation.calculations import latlon_distance
    
    plt.style.use("Plots/ctuthesis.mplstyle")
    
    # Get unique satellite IDs
    sat_ids = np.unique(curve_array[:, IDX.sat_id])
    
    # Create color picker for different satellites
    color_picker = ColorPicker()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])  # Full width for time series
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom left for scatter (estimated)
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom right for scatter (true)
    ax4 = fig.add_subplot(gs[2, :])  # Full width for residuals comparison
    
    # User positions
    r_est = (EarthLocation.from_geodetic(lon_est, lat_est, alt_est)
             .get_itrs().cartesian.without_differentials())
    ru_est = np.array([r_est.x.to("km").value, 
                       r_est.y.to("km").value, 
                       r_est.z.to("km").value]) * 1000  # Convert to meters
    
    r_true = (EarthLocation.from_geodetic(lon_true, lat_true, alt_true)
              .get_itrs().cartesian.without_differentials())
    ru_true = np.array([r_true.x.to("km").value, 
                        r_true.y.to("km").value, 
                        r_true.z.to("km").value]) * 1000  # Convert to meters
    
    C = 299792458  # m/s
    pos_error = latlon_distance(lat_true, lat_est, lon_true, lon_est, alt_true, alt_est)
    
    # Plot data for each satellite
    for sat_id in sat_ids:
        # Filter data for this satellite
        sat_mask = curve_array[:, IDX.sat_id] == sat_id
        sat_data = curve_array[sat_mask]
        
        if len(sat_data) == 0:
            continue
            
        # Extract data
        times = sat_data[:, IDX.t]
        measured_freq = sat_data[:, IDX.f]
        base_freq = sat_data[:, IDX.fb]
        measured_doppler = measured_freq - base_freq
        
        # Satellite positions and velocities
        rs = np.array([sat_data[:, IDX.x], sat_data[:, IDX.y], sat_data[:, IDX.z]]) * 1000  # km to m
        vs = np.array([sat_data[:, IDX.vx], sat_data[:, IDX.vy], sat_data[:, IDX.vz]]) * 1000  # km/s to m/s
        
        # Calculate predicted doppler for ESTIMATED position
        ru_est_tiled = np.tile(ru_est.reshape(3, 1), (1, len(times)))
        rel_vel_est = np.sum(vs * (rs - ru_est_tiled) / np.linalg.norm(rs - ru_est_tiled, axis=0), axis=0)
        predicted_doppler_est = -1 * rel_vel_est * base_freq / C
        drift_est = (times - np.min(times)) * dft_est
        predicted_doppler_est += off_est + drift_est
        
        # Calculate predicted doppler for TRUE position (assume offset and drift from estimated)
        ru_true_tiled = np.tile(ru_true.reshape(3, 1), (1, len(times)))
        rel_vel_true = np.sum(vs * (rs - ru_true_tiled) / np.linalg.norm(rs - ru_true_tiled, axis=0), axis=0)
        predicted_doppler_true = -1 * rel_vel_true * base_freq / C
        predicted_doppler_true += off_est + drift_est  # Use same offset/drift for fair comparison
        
        # Residuals
        residuals_est = measured_doppler - predicted_doppler_est
        residuals_true = measured_doppler - predicted_doppler_true
        
        # Get satellite name
        sat_name = f"Sat {int(sat_id)}"
        if satellites is not None:
            try:
                sat_obj = satellites.get(str(int(sat_id)))
                if sat_obj:
                    sat_name = f"{sat_obj.name} ({int(sat_id)})"
            except:
                pass
        
        # Pick color for this satellite
        color = color_picker.pick_color(sat_name)
        
        # Plot 1: Time series - measured, predicted (est), predicted (true)
        ax1.scatter(times, measured_doppler, marker="o", s=20, color=color, alpha=0.6, label=f"{sat_name} (meas)")
        ax1.plot(times, predicted_doppler_est, '--', linewidth=2, color=color, alpha=0.5, label=f"{sat_name} (est)")
        ax1.plot(times, predicted_doppler_true, ':', linewidth=3, color=color, alpha=0.9, label=f"{sat_name} (TRUE)")
        
        # Plot 2: Scatter - estimated position
        ax2.scatter(predicted_doppler_est, measured_doppler, marker=".", s=20, color=color, alpha=0.6)
        
        # Plot 3: Scatter - true position
        ax3.scatter(predicted_doppler_true, measured_doppler, marker=".", s=20, color=color, alpha=0.6, label=sat_name)
        
        # Plot 4: Residuals comparison
        ax4.scatter(times, residuals_est, marker="x", s=20, color=color, alpha=0.4, label=f"{sat_name} (est)")
        ax4.scatter(times, residuals_true, marker="o", s=15, color=color, alpha=0.7)
    
    # Configure plot 1 (time series)
    ax1.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.3)
    ax1.set_xlabel("Time [Unix seconds]", fontsize=10)
    ax1.set_ylabel("Doppler Shift [Hz]", fontsize=10)
    ax1.set_title(f"Measured vs Predicted Doppler\n" +
                  f"○ Measured | - - Predicted (EST: {lat_est:.2f}°, {lon_est:.2f}°) | ··· Predicted (TRUE: {lat_true:.2f}°, {lon_true:.2f}°)\n" +
                  f"Position Error: {pos_error:.0f} m", 
                  fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=5, ncol=3, markerscale=0.5)
    
    # Configure plot 2 (scatter - estimated)
    all_doppler = np.concatenate([curve_array[:, IDX.f] - curve_array[:, IDX.fb]])
    diag_min, diag_max = all_doppler.min(), all_doppler.max()
    ax2.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', linewidth=2, alpha=0.5)
    ax2.set_xlabel("Predicted Doppler (Estimated Pos) [Hz]", fontsize=10)
    ax2.set_ylabel("Measured Doppler [Hz]", fontsize=10)
    ax2.set_title(f"Estimated Position\n({lat_est:.4f}°, {lon_est:.4f}°, {alt_est:.0f}m)", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # Configure plot 3 (scatter - true)
    ax3.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', linewidth=2, alpha=0.5)
    ax3.set_xlabel("Predicted Doppler (True Pos) [Hz]", fontsize=10)
    ax3.set_ylabel("Measured Doppler [Hz]", fontsize=10)
    ax3.set_title(f"True Position\n({lat_true:.4f}°, {lon_true:.4f}°, {alt_true:.0f}m)", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=7, ncol=1, markerscale=0.8)
    ax3.set_aspect('equal', adjustable='box')
    
    # Configure plot 4 (residuals comparison)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax4.set_xlabel("Time [Unix seconds]", fontsize=10)
    ax4.set_ylabel("Residuals [Hz]", fontsize=10)
    ax4.set_title("Residuals: × Estimated Position | ○ True Position (better = closer to 0)", fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=6, ncol=3, markerscale=0.8)
    
    plt.savefig(get_fig_filename("doppler_comparison_est_vs_true"))
    
    if show:
        plt.show()


def plot_measured_vs_predicted_doppler(curve_array, lat, lon, alt, off, dft, satellites=None, show=False):
    """
    Plot measured vs predicted Doppler for all satellites
    
    :param curve_array: array with columns [t, f, fb, sat_id, x, y, z, vx, vy, vz, ...]
    :param lat: final latitude
    :param lon: final longitude  
    :param alt: final altitude
    :param off: final offset
    :param dft: final drift
    :param satellites: dictionary of satellite objects (for names)
    :param show: whether to show the plot
    """
    from astropy.coordinates import EarthLocation
    from src.navigation.data_processing import NavDataArrayIndices as IDX
    
    plt.style.use("Plots/ctuthesis.mplstyle")
    
    # Get unique satellite IDs
    sat_ids = np.unique(curve_array[:, IDX.sat_id])
    
    # Create color picker for different satellites
    color_picker = ColorPicker()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # User position
    r_user_arr = (EarthLocation.from_geodetic(lon, lat, alt)
                  .get_itrs().cartesian.without_differentials())
    ru = np.array([r_user_arr.x.to("km").value, 
                   r_user_arr.y.to("km").value, 
                   r_user_arr.z.to("km").value]) * 1000  # Convert to meters
    
    C = 299792458  # m/s
    
    # Plot measured vs predicted for each satellite
    for sat_id in sat_ids:
        # Filter data for this satellite
        sat_mask = curve_array[:, IDX.sat_id] == sat_id
        sat_data = curve_array[sat_mask]
        
        if len(sat_data) == 0:
            continue
            
        # Extract data
        times = sat_data[:, IDX.t]
        measured_freq = sat_data[:, IDX.f]
        base_freq = sat_data[:, IDX.fb]
        measured_doppler = measured_freq - base_freq
        
        # Satellite positions and velocities
        rs = np.array([sat_data[:, IDX.x], sat_data[:, IDX.y], sat_data[:, IDX.z]]) * 1000  # km to m
        vs = np.array([sat_data[:, IDX.vx], sat_data[:, IDX.vy], sat_data[:, IDX.vz]]) * 1000  # km/s to m/s
        
        # Calculate predicted doppler for this satellite
        ru_tiled = np.tile(ru.reshape(3, 1), (1, len(times)))
        rel_vel = np.sum(vs * (rs - ru_tiled) / np.linalg.norm(rs - ru_tiled, axis=0), axis=0)
        predicted_doppler = -1 * rel_vel * base_freq / C
        
        # Add offset and drift
        drift = (times - np.min(times)) * dft
        predicted_doppler += off + drift
        
        # Get satellite name
        sat_name = f"Sat {int(sat_id)}"
        if satellites is not None:
            try:
                sat_obj = satellites.get(str(int(sat_id)))
                if sat_obj:
                    sat_name = f"{sat_obj.name} ({int(sat_id)})"
            except:
                pass
        
        # Pick color for this satellite
        color = color_picker.pick_color(sat_name)
        
        # Plot measured (solid line) and predicted (dashed line) on top plot
        ax1.scatter(times, measured_doppler, marker=".", s=15, color=color, alpha=0.7, label=f"{sat_name} (measured)")
        ax1.plot(times, predicted_doppler, '--', linewidth=2, color=color, alpha=0.9)
        
        # Plot measured vs predicted scatter on bottom plot
        ax2.scatter(predicted_doppler, measured_doppler, marker=".", s=15, color=color, alpha=0.6, label=sat_name)
    
    # Configure top plot (time series)
    ax1.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.3)
    ax1.set_xlabel("Time [Unix seconds]", fontsize=11)
    ax1.set_ylabel("Doppler Shift [Hz]", fontsize=11)
    ax1.set_title(f"Measured vs Predicted Doppler (Solid = Measured, Dashed = Predicted)\n" +
                  f"Position: ({lat:.4f}°, {lon:.4f}°, {alt:.0f}m), Offset: {off:.1f} Hz, Drift: {dft:.3f} Hz/s", 
                  fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=6, ncol=2, markerscale=0.5)
    
    # Configure bottom plot (measured vs predicted scatter)
    # Add diagonal line (y=x) which represents perfect agreement
    all_doppler = np.concatenate([curve_array[:, IDX.f] - curve_array[:, IDX.fb]])
    diag_min, diag_max = all_doppler.min(), all_doppler.max()
    ax2.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', linewidth=2, alpha=0.5, label='Perfect fit (y=x)')
    ax2.set_xlabel("Predicted Doppler [Hz]", fontsize=11)
    ax2.set_ylabel("Measured Doppler [Hz]", fontsize=11)
    ax2.set_title("Measured vs Predicted Scatter Plot (Points should lie on diagonal)", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=7, ncol=2, markerscale=0.5)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(get_fig_filename("measured_vs_predicted_doppler"))
    
    if show:
        plt.show()


def plot_residuals_per_satellite(curve_array, lat, lon, alt, off, dft, satellites=None, show=False):
    """
    Plot residuals (Measured Freq - Predicted Freq) per satellite to identify ghost satellites
    
    :param curve_array: array with columns [t, f, fb, sat_id, x, y, z, vx, vy, vz, ...]
    :param lat: final latitude
    :param lon: final longitude  
    :param alt: final altitude
    :param off: final offset
    :param dft: final drift
    :param satellites: dictionary of satellite objects (for names)
    :param show: whether to show the plot
    """
    from astropy.coordinates import EarthLocation
    from src.navigation.data_processing import NavDataArrayIndices as IDX
    
    plt.style.use("Plots/ctuthesis.mplstyle")
    
    # Get unique satellite IDs
    sat_ids = np.unique(curve_array[:, IDX.sat_id])
    
    # Create color picker for different satellites
    color_picker = ColorPicker()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # User position
    r_user_arr = (EarthLocation.from_geodetic(lon, lat, alt)
                  .get_itrs().cartesian.without_differentials())
    ru = np.array([r_user_arr.x.to("km").value, 
                   r_user_arr.y.to("km").value, 
                   r_user_arr.z.to("km").value]) * 1000  # Convert to meters
    
    C = 299792458  # m/s
    
    residual_stats = []
    
    for sat_id in sat_ids:
        # Filter data for this satellite
        sat_mask = curve_array[:, IDX.sat_id] == sat_id
        sat_data = curve_array[sat_mask]
        
        if len(sat_data) == 0:
            continue
            
        # Extract data
        times = sat_data[:, IDX.t]
        measured_freq = sat_data[:, IDX.f]
        base_freq = sat_data[:, IDX.fb]
        measured_doppler = measured_freq - base_freq
        
        # Satellite positions and velocities
        rs = np.array([sat_data[:, IDX.x], sat_data[:, IDX.y], sat_data[:, IDX.z]]) * 1000  # km to m
        vs = np.array([sat_data[:, IDX.vx], sat_data[:, IDX.vy], sat_data[:, IDX.vz]]) * 1000  # km/s to m/s
        
        # Calculate predicted doppler for this satellite
        ru_tiled = np.tile(ru.reshape(3, 1), (1, len(times)))
        rel_vel = np.sum(vs * (rs - ru_tiled) / np.linalg.norm(rs - ru_tiled, axis=0), axis=0)
        predicted_doppler = -1 * rel_vel * base_freq / C
        
        # Add offset and drift
        drift = (times - np.min(times)) * dft
        predicted_doppler += off + drift
        
        # Calculate residuals
        residuals = measured_doppler - predicted_doppler
        
        # Get satellite name
        sat_name = f"Sat {int(sat_id)}"
        if satellites is not None:
            try:
                sat_obj = satellites.get(str(int(sat_id)))
                if sat_obj:
                    sat_name = f"{sat_obj.name} ({int(sat_id)})"
            except:
                pass
        
        # Calculate statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        rms_residual = np.sqrt(np.mean(residuals**2))
        max_residual = np.max(np.abs(residuals))
        
        residual_stats.append({
            'sat_id': int(sat_id),
            'name': sat_name,
            'mean': mean_residual,
            'std': std_residual,
            'rms': rms_residual,
            'max': max_residual,
            'n_points': len(residuals)
        })
        
        # Pick color for this satellite
        color = color_picker.pick_color(sat_name)
        
        # Plot residuals vs time
        ax1.scatter(times, residuals, marker=".", s=10, label=sat_name, color=color, alpha=0.6)
        
        # Plot histogram of residuals
        ax2.hist(residuals, bins=30, alpha=0.5, label=sat_name, color=color)
    
    # Configure residuals vs time plot
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel("Time [Unix seconds]")
    ax1.set_ylabel("Residuals (Measured - Predicted) [Hz]")
    ax1.set_title(f"Residuals per Satellite\nPosition: ({lat:.4f}°, {lon:.4f}°, {alt:.0f}m), Offset: {off:.1f} Hz, Drift: {dft:.3f} Hz/s")
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Configure histogram plot
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel("Residuals [Hz]")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Residuals")
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(get_fig_filename("residuals_per_satellite"))
    
    # Print statistics
    print("\n" + "="*80)
    print("RESIDUAL ANALYSIS - Ghost Satellite Detection")
    print("="*80)
    print(f"{'Satellite':<25} {'Mean [Hz]':>12} {'Std [Hz]':>12} {'RMS [Hz]':>12} {'Max [Hz]':>12} {'N Points':>10}")
    print("-"*80)
    
    # Sort by RMS (worst first)
    residual_stats.sort(key=lambda x: x['rms'], reverse=True)
    
    for stat in residual_stats:
        warning = "  *** GHOST? ***" if stat['rms'] > 1000 else ""  # Flag satellites with >1kHz RMS
        print(f"{stat['name']:<25} {stat['mean']:>12.1f} {stat['std']:>12.1f} {stat['rms']:>12.1f} {stat['max']:>12.1f} {stat['n_points']:>10}{warning}")
    
    print("="*80)
    print("Note: Satellites with RMS residuals >1kHz may have incorrect IDs (Ghost satellites)")
    print("Good fit: Residuals should hover around 0 Hz with random noise")
    print("Bad fit: Clear trends or large systematic bias indicate wrong satellite ID")
    print("="*80 + "\n")
    
    if show:
        plt.show()
    
    return residual_stats


if __name__ == "__main__":
    for i in [1104]:
        plot_results_of_iterative_position_finding(f"results_{i}")
    plt.show()
