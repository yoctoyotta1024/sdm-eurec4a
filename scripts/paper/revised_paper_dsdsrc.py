# %%
def get_mfdataset(datapath, microphysics, valid_cloud_ids=None, normheight=False):
  import glob
  import xarray as xr
  from pathlib import Path

  if normheight:
    ds_name = "droplet_pdfdistribs_normheight.zarr"
  else:
    ds_name = "droplet_pdfdistribs.zarr"
 
  datasets = sorted(glob.glob(str(datapath / microphysics / "cluster_*" / "processed" / ds_name)))
  cluster_names = [Path(d).parent.parent.name for d in datasets]

  if valid_cloud_ids is not None:
    valid_cloud_id_names = ["cluster_"+str(x) for x in valid_cloud_ids]
    ds_name_pairs = [
        (ds, name)
        for ds, name in zip(datasets, cluster_names)
        if name in valid_cloud_id_names
    ]
    datasets = [dataset for dataset, _ in ds_name_pairs]
    cluster_names = [cluster_name for _, cluster_name in ds_name_pairs]

  print(f"{len(cluster_names)} valid clusters found with {ds_name}:\n", cluster_names)

  ds = xr.open_mfdataset(datasets,
                           engine="zarr",
                           combine="nested",
                           concat_dim="cluster"
                           )

  def get_number_after_underscore(s):
    parts = s.split("_")
    return int(parts[-1]) if len(parts) > 1 else None

  cluster_nums = dict(cluster=("cluster", [get_number_after_underscore(c) for c in cluster_names]))
  ds = ds.assign_coords(cluster_nums)
  
  ds.attrs = {
     "microphysics":  microphysics,
     "normalised_by_height": normheight
     }

  return ds

# %%
def sanity_check_plot(datasets, microphysics_styles, extra_ds, clusters2plot=None, normheight=False):
    import matplotlib.pyplot as plt
    import numpy as np

    print("Distributions at Top and Bottom of Domain for all Microphysics Setups")

    default_ds = datasets["condensation"]
    if clusters2plot is None:
       clusters2plot = default_ds.cluster.values


    for cluster in clusters2plot:
        if normheight:
            heights2plot = [1.0, 0.0]
        else:
            coord3 = extra_ds.gridbox_coord3.sel(microphysics="condensation", cloud_id=cluster)
            heights2plot = [coord3.max(), coord3.min()]

        ncols=len(datasets.keys())
        nrows=len(heights2plot)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,5))
        fig.suptitle(f"cluster {cluster}")

        for i, h in enumerate(heights2plot):
            axs = axes[i,:]
            if normheight:
                text = f"level={h:.0f}"
            else:
                text = f"{h:.0f}m"
            axs[0].text(
                0.02,
                0.98,
                text,
                transform=axs[0].transAxes,
                ha="left",
                va="top",
            )
            for a, (microphys, ds) in enumerate(datasets.items()):
                ax = axs[a]
                if microphys == "condensation_novent":
                   title = microphysics_styles["condensation"]["name"]
                else:
                   title = microphysics_styles[microphys]["name"]
                ax.set_title(title)

                if normheight:
                    # data = ds.widths.values*ds.nsupers_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
                    # data = ds.numconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
                    data = ds.massconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
                else:
                    # data = ds.widths.values*ds.nsupers_pdf.sel(cluster=cluster).sel(height=h, method="nearest")
                    # data = ds.numconc_pdf.sel(cluster=cluster).sel(height=h, method="nearest")
                    data = ds.massconc_pdf.sel(cluster=cluster).sel(height=h, method="nearest")
                ax.step(ds.centers, data, where="mid")

        for ax in axes.flatten():
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim([1e-1, 5e4])
        for ax in axes[-1,:]:
            ax.set_xlabel("radius / microns")
        for ax in axes[:,0]:
            # ax.set_ylabel("nsupers")
            # ax.set_ylabel("numconc / cm^-3 m^-1")
            ax.set_ylabel("massconc / g m^-3 m^-1")

        plt.tight_layout()
        plt.show()


def select_collisions_diff_clouds(ds_normalized, microphysics_styles):
    import matplotlib.pyplot as plt
    import numpy as np

    plot_limits = {
        "collision_condensation" : 17,
        "coalbure_condensation_small": 32,
        "coalbure_condensation_large": 975,
   }
    selected_clusters = {
        "collision_condensation" : [],
        "coalbure_condensation_small": [],
        "coalbure_condensation_large": [],
   }

    x_all = -ds_normalized["evaporation_rate_energy"]
    x_refernce = x_all.sel(microphysics="condensation")
    attrs = x_all.attrs.copy()
    x = (x_all - x_refernce) / x_refernce * 100
    x.attrs.update(
        long_name=f"Relative difference of {attrs['long_name']} compared to {microphysics_styles['condensation']['name']}",
        units=r"\%",
    )
    y = ds_normalized["normalized_gridbox_coord3"]

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(5, 8), sharey=True)
    
    plot_microphysics = [
        "collision_condensation",
        "coalbure_condensation_small",
        "coalbure_condensation_large",
    ]
    for _ax, mp in zip(axs, plot_microphysics):

        style_full = microphysics_styles[mp].copy()
        _ax.set_title(microphysics_styles.get_setup(mp)["name"])

        for id in ds_normalized.cloud_id:
            x_cloud = x.sel(microphysics=mp, cloud_id=id)
            if mp == "collision_condensation":
                compare = -x_cloud
            else:
                compare = x_cloud
            if np.any(compare > plot_limits[mp]):
                color=style_full["dark_color"]
                selected_clusters[mp].append(id.values)
            else:
                color="grey"
            _ax.plot(
                x_cloud,
                y,
                color=color,
                linestyle="--",
                label="Mean",
                zorder=10,
            )

    for _ax in axs:
        _ax.axvline(0, color="k", linestyle="--", alpha=0.5, zorder=10)
        _ax.set_ylim(0, 1)
        _ax.set_yticks([0, 0.5, 1])

    axs[0].set_xlim(-25, 10)
    axs[1].set_xlim(-15, 80)
    axs[2].set_xlim(-15, 1100)

    for ax in axs:
        ax.set_ylabel("Normalized Height []")
        ax.set_xlabel("Rel Diff. to EvapOnly [%]")

    fig.tight_layout()

    for key, value in selected_clusters.items():
        selected_clusters[key] = np.asarray(value)

    return fig, selected_clusters