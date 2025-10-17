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
        