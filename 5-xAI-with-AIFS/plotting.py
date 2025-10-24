import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np


def plot_sensitivities(
    state: dict, field: str, savefig: bool = False, area: tuple[float, float, float, float] = None
) -> None:
    num_times = state["fields"][field].shape[0]
    fig, axs = plt.subplots(1, num_times, figsize=(16, 12 * num_times), subplot_kw={'projection': ccrs.PlateCarree()})
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    # Get the combined min/max for color normalization
    lim = max(abs(state["fields"][field].min()), abs(state["fields"][field].max()))

    scatter_plots = []
    for i in range(num_times):
        ax = axs[i]
        ax.set_title(f"{field} (at -{(num_times-i-1)*6}H)")
        ax.add_feature(cfeature.COASTLINE, alpha=0.8, color="grey")
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.4, color="grey")
        if area is not None:
            # area should be (lon_min, lon_max, lat_min, lat_max)
            ax.set_extent(area, crs=ccrs.PlateCarree())

        sensitivities = state["fields"][field][i]
        scatter = ax.scatter(
            state["longitudes"],
            state["latitudes"],
            c=sensitivities,
            cmap="PuOr",
            vmin=-lim,
            vmax=lim,
            transform=ccrs.PlateCarree()
        )
        scatter_plots.append(scatter)
    
    # Use the first scatter plot for the colorbar to ensure colormap consistency
    cbar = fig.colorbar(
        scatter_plots[0],
        ax=axs,
        orientation='horizontal',
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


def plot_summary_pl(stats_df, stats: list[str], cmaps: dict[str, str] = None, savefig: bool = False):
    if cmaps is None or isinstance(cmaps, str):
        cmaps = {s: "viridis" if cmaps is None else cmaps for s in stats}
    
    hours = list(stats_df["hour"].unique())
    
    fig, axs = plt.subplots(
        len(hours),
        len(stats),
        figsize=(8 * len(stats), 5 * len(hours)),
        sharex=True,
        sharey=True
    )
    
    # Track pcolormesh handles for each stat to normalize color scales
    pcms = {stat: [] for stat in stats}
    
    for i, hour in enumerate(hours):
        for j, stat in enumerate(stats):
            # Filter and pivot your DataFrame
            heatmap_data = (
                stats_df[(stats_df.type == "pl") & (stats_df.hour == hour)]
                .astype({"pl": int})
                .set_index(["varname", "pl"])[stat]
                .sort_index()
                .unstack()
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
                axs[i, j].set_xlabel("Pressure Level")
            if j == 0:
                axs[i, j].set_ylabel("Variable (only pl)")
    
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
        plt.xticks(rotation=45)

    if savefig:
        fig.savefig(f"sensitivities_pl_summary.png")
        plt.close(fig)


def plot_summary_sfc(stats_df, stats: list[str], cmaps: dict[str, str] = None, savefig: bool = False):
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
