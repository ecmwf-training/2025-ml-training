import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)

def fix(lons):
    # Shift the longitudes from 0-360 to -180-180
    return np.where(lons > 180, lons - 360, lons)
    
def plot_sensitivities(
    state: dict, field: str, savefig: bool = False, area: tuple[float, float, float, float] = None, size: float | None = None,
) -> None:
    """Plot sensitivities on a map for a given field."""
    num_times = state["fields"][field].shape[0]
    fig, axs = plt.subplots(1, num_times, figsize=(16, 12 * num_times), subplot_kw={"projection": ccrs.PlateCarree()})
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    # Get the combined min/max for color normalization
    lim = max(abs(state["fields"][field].min()), abs(state["fields"][field].max()))

    scatter_plots = []
    for i in range(num_times):
        ax = axs[i]
        ax.set_title(f"{field} (at -{(num_times-i-1)*6}H)")
        ax.add_feature(cfeature.COASTLINE, alpha=0.8, color="grey")
        ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.4, color="grey")
        if area is not None:
            # area should be (lon_min, lon_max, lat_min, lat_max)
            ax.set_extent(area, crs=ccrs.PlateCarree())

        sensitivities = state["fields"][field][i]
        scatter = ax.tricontourf(
            fix(state["longitudes"]),
            state["latitudes"],
            sensitivities,
            cmap="PuOr",
            vmin=-lim,
            vmax=lim,
            transform=ccrs.PlateCarree(),
        )
        scatter_plots.append(scatter)

    # Use the first scatter plot for the colorbar to ensure colormap consistency
    cbar = fig.colorbar(
        scatter_plots[0],
        ax=axs,
        orientation="horizontal",
        fraction=0.05,
        shrink=0.7,
        pad=0.05,
    )
    cbar.set_label(f"{field} sensitivity")

    # Remove x and y axes
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    if savefig:
        fig.savefig(f"sensitivities_{field}.png")
        plt.close(fig)


def plot_summary_pl(
    stats_df: pd.DataFrame, 
    stats: list[str],
    variables: list[str] | None = None,
    cmaps: dict[str, str] | None = None,
    savefig: bool = False
) -> None:
    """Plot summary statistics for pressure levels."""
    if cmaps is None or isinstance(cmaps, str):
        cmaps = {s: "viridis" if cmaps is None else cmaps for s in stats}
    
    hours = list(stats_df["hour"].unique())
    variables = list(stats_df[stats_df.type == "pl"]["varname"].unique()) if variables is None else variables
    
    fig, axs = plt.subplots(
        len(hours),
        len(stats),
        figsize=(5 * len(stats), 8 * len(hours)),
        sharex=True,
        sharey=True
    )
    
    # Track pcolormesh handles for each stat to normalize color scales
    pcms = {stat: [] for stat in stats}
    
    for i, hour in enumerate(hours):
        for j, stat in enumerate(stats):
            # Filter and pivot your DataFrame
            heatmap_data = (
                stats_df[(stats_df.type == "pl") & (stats_df.hour == hour) & (stats_df.varname.isin(variables))]
                .astype({"pl": int})
                .set_index(["varname", "pl"])[stat]
                .unstack().T
                .sort_index(ascending=False)
            )

            # Plot heatmap
            pcm = axs[i, j].pcolormesh(
                heatmap_data.values,
                cmap=cmaps[stat],
                shading="auto",
                edgecolors='none',
            )
            pcms[stat].append(pcm)
    
            # Axis ticks and labels
            axs[i, j].set_xticks(np.arange(len(heatmap_data.columns)) + 0.5)
            axs[i, j].set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
            axs[i, j].set_xticklabels(heatmap_data.columns)
            axs[i, j].set_yticklabels(heatmap_data.index)
            axs[i, j].grid(False)
            axs[i, j].set_frame_on(True)
    
            # Axis labels only on edges
            if i == len(hours) - 1:
                axs[i, j].set_xlabel("Variable (only pl)")
            if j == 0:
                axs[i, j].set_ylabel("Pressure Level")
    
            axs[i, j].set_title(f"{stat.upper()} values at {hour}h")
    
    # --- Shared colorbars (one per column, below each column) ---
    for j, stat in enumerate(stats):
        # Determine color scale across all rows for this stat
        vmin = min(pcm.get_array().min() for pcm in pcms[stat])
        vmax = max(pcm.get_array().max() for pcm in pcms[stat])
        for pcm in pcms[stat]:
            pcm.set_clim(vmin, vmax)
    
        # Add horizontal colorbar below each column
        cbar = fig.colorbar(
            pcms[stat][0],
            ax=axs[:, j],          # all rows in column j
            orientation="horizontal",
            fraction=0.05, pad=0.15
        )
        cbar.set_label(f"{stat.upper()} value")
    
    # Rotate x-tick labels
    for ax in axs.flat:
        plt.sca(ax)

    if savefig:
        fig.savefig(f"sensitivities_pl_summary.png")
        plt.close(fig)


def plot_summary_sfc(
    stats_df: pd.DataFrame,
    stats: list[str],
    cmaps: dict[str, str] | None = None,
    savefig: bool = False
) -> None:
    """Plot summary statistics for surface variables."""
    if cmaps is None or isinstance(cmaps, str):
        cmaps = {s: "viridis" if cmaps is None else cmaps for s in stats}

    hours = list(stats_df["hour"].unique())

    fig, axs = plt.subplots(
        len(stats),
        1,
        figsize=(12, 2.5  * len(stats)),
        sharex=True,
        sharey=True
    )
    
    # Track pcolormesh handles for each stat to normalize color scales
    pcms = {stat: [] for stat in stats}
    
    for j, stat in enumerate(stats):
        # Filter and pivot your DataFrame
        heatmap_data = (
            stats_df[(stats_df.type == "sfc")]
            .set_index(["varname", "hour"])[stat]
            .unstack()
            .T
        )
    
        # Plot heatmap
        pcm = axs[j].pcolormesh(
            heatmap_data.values,
            cmap=cmaps[stat],
            shading="auto",
            edgecolors='none',
        )
        pcms[stat].append(pcm)
    
        # Axis ticks and labels
        axs[j].set_xticks(np.arange(len(heatmap_data.columns)) + 0.5)
        axs[j].set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
        axs[j].set_xticklabels(heatmap_data.columns)
        axs[j].set_yticklabels(heatmap_data.index)
        axs[j].grid(False)
        axs[j].set_frame_on(True)

        # Axis labels only on edges
        if j == len(stats) - 1:
            axs[j].set_xlabel("Pressure Level")
        axs[j].set_ylabel("Input hour")
    
        if j == 0:
            axs[j].set_title(f"Sensitivities of SFC variables")
    
    for j, stat in enumerate(stats):
        # Determine color scale across all rows for this stat
        vmin = min(pcm.get_array().min() for pcm in pcms[stat])
        vmax = max(pcm.get_array().max() for pcm in pcms[stat])
        for pcm in pcms[stat]:
            pcm.set_clim(vmin, vmax)
    
        # Add horizontal colorbar below each column
        cbar = fig.colorbar(
            pcms[stat][0],
            ax=axs[j],          # all rows in column j
            orientation="vertical",
            fraction=0.05, pad=0.15
        )
        cbar.set_label(f"{stat.upper()} value")
    
    # Rotate x-tick labels
    for ax in axs:
        plt.sca(ax)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    if savefig:
        fig.savefig(f"sensitivities_sfc_summary.png")
        plt.close(fig)


def plot_cross_section(
    variable: str,
    state: dict,
    latitude: float | None= None,
    longitude: float | None = None,
    margin: float = 0.2,
    xlim: tuple[float, float] | None = None,
    savefig: bool = False,
) -> None:
    """Plot latitude cross-section subplots for a given variable."""
    assert (latitude is not None) != (longitude is not None), "Specify either latitude or longitude."
    latitudes = state["latitudes"]
    longitudes = state["longitudes"]
    data = state["fields"]
    if longitude is not None:
        ref_value = longitude
        ref_coords = longitudes
        xvalues = latitudes
        xlabel = "Latitude"
    else:
        ref_value = latitude
        ref_coords = latitudes
        xvalues = longitudes
        xlabel = "Longitude"

    mask = (ref_coords >= ref_value - margin) & (ref_coords <= ref_value + margin)
    if mask.sum() == 0:
        raise ValueError(
            f"No data points found in the {xlabel.lower()} interval ({ref_value - margin}, {ref_value + margin})."
        )

    # Get pressure levels
    pls = list(reversed(sorted(int(k.split("_")[1]) for k in data.keys() if k.startswith(f"{variable}_"))))
    print(
        f"Plotting cross-section sensitivities at {xlabel.lower()}={ref_value}° with {mask.sum()} points at {len(pls)} pressure levels."
    )

    # Collect Ys for each pressure level and each data slice (e.g., data[0], data[1])
    sensitivities = np.array([data[f"{variable}_{pl}"][:, mask] for pl in pls]).transpose(1, 0, 2)
    # shape: [input_steps, num_pls, num_gridpoints]

    # Prepare cmap limits
    vmax = np.abs(sensitivities).max()

    # Create meshgrid for plotting
    Xs, Ys = np.meshgrid(xvalues[mask], pls)

    if xlim is None:
        xlim = xvalues[mask].min(), xvalues[mask].max()

    # Compute x-axis bounds
    fig, axs = plt.subplots(1, sensitivities.shape[0], figsize=(8 * sensitivities.shape[0], 5), sharey=True)

    for i, ax in enumerate(axs):
        pcm = ax.imshow(
            sensitivities[i],
            extent=[Xs.min(), Xs.max(), Ys.min(), Ys.max()],
            vmin=-vmax,
            vmax=vmax,
            cmap="RdBu",
            aspect='auto'
        )
        ax.set_xlim(xlim)
        ax.set_title(f"{variable}[-{(len(axs)-i-1)*6}H] sensitivity at {xlabel[:3].lower()}={ref_value}°")
        ax.set_xlabel(f"{xlabel} (°)")
        if i == 0:
            ax.set_ylabel("Pressure Level (hPa)")

    ax.invert_yaxis()
    fig.colorbar(pcm, ax=ax, label=f"{variable}")

    plt.tight_layout()
    if savefig:
        fig.savefig(f"sensitivities_{variable}_{xlabel[:3].lower()}{ref_value}.png")
        plt.close(fig)
