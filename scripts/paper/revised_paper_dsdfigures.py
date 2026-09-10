# %% Input Settings
import numpy as np
import matplotlib.pyplot as plt

from importlib import reload
from pathlib import Path
from ruamel.yaml import YAML
from sdm_eurec4a import RepositoryPath, data_loading
from sdm_eurec4a.visulization import (
    save_figure,
)

import revised_paper_figures_src as src
import revised_paper_dsdsrc as dsdplots
RepoPaths = RepositoryPath("levante_m300950")

data_dir = RepoPaths.CLEO_data_dir / "output_v4.2"
data_dir_novent = RepoPaths.CLEO_data_dir / "output_v4.2_novent"
data_dir_obs = RepoPaths.data_dir / "observation"

fig_dir_2 = RepoPaths.fig_dir / Path("paper-revised")
fig_dir = fig_dir_2 / Path("dsd_plots")
fig_dir.mkdir(exist_ok=True, parents=False)

yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
valid_cloud_ids_file = data_dir / Path("valid_cloud_ids.yaml")
valid_cloud_ids = sorted(yaml.load(valid_cloud_ids_file)["valid_cloud_ids"])

print(f"Using data from {data_dir}")
print(f"Using novent data from: {data_dir_novent}")
print(f"Using obs data from {data_dir_obs}")
print(f"using {len(valid_cloud_ids)} valid_cloud_ids from: {valid_cloud_ids_file}")
# %% Load datasets, takes circa. 2mins
microphysics_styles = data_loading.MicrophysicsStyles()
microphysics_datasets = {
   m: dsdplots.get_mfdataset(data_dir, m, valid_cloud_ids=valid_cloud_ids, normheight=False) for m in tuple(microphysics_styles)
}
microphysics_datasets["condensation_novent"] = dsdplots.get_mfdataset(data_dir_novent, "condensation", normheight=False)
microphysics_datasets
# %% Load datasets, takes circa. 2mins
microphysics_datasets_norm = {
   m: dsdplots.get_mfdataset(data_dir, m, valid_cloud_ids=valid_cloud_ids, normheight=True) for m in tuple(microphysics_styles)
}
microphysics_datasets_norm["condensation_novent"] = dsdplots.get_mfdataset(data_dir_novent, "condensation", normheight=True)
microphysics_datasets_norm
# %% Load Extra Datasets
_, _, extra_ds, \
    extra_ds_sem, \
    extra_ds_normalized, \
    extra_ds_normalized_sem, \
    extra_ds_no_ventilation, \
    extra_ds_sem_no_ventilation, \
    _, _, _ = src.load_required_datasets_and_remove_outliers(data_dir, \
                                                                      data_dir_novent, \
                                                                        data_dir_obs, \
                                                                          microphysics_styles)
# %%
clusters2plot = np.random.choice(microphysics_datasets["condensation"].cluster.values, size=10, replace=False)
# sanity_check_plot(microphysics_datasets_norm, microphysics_styles, extra_ds, clusters2plot=clusters2plot, normheight=True)
dsdplots.sanity_check_plot(microphysics_datasets, microphysics_styles, extra_ds, clusters2plot=clusters2plot)
plt.show()
# %% Select which clouds to plot for example DSDs of effect of collisions
fig, collisions_selected_clusters = dsdplots.select_collisions_diff_clouds(extra_ds_normalized, microphysics_styles)
save_figure(fig=fig, filepath=fig_dir / "dsd_effect_breakup_selection_criteria")
plt.show()
collisions_selected_clusters
# %%
fig = dsdplots.plot_collision_effect_evaporation_dsds(microphysics_datasets_norm, collisions_selected_clusters, microphysics_styles)
save_figure(fig=fig, filepath=fig_dir / "dsd_effect_breakup")
plt.show()
# %% Select which clouds to plot for example DSDs of condensation setup
fig, condonly_selected_clusters, condonly_selected_clusters_colors = dsdplots.select_condonly_diff_clouds(extra_ds_normalized)
save_figure(fig=fig, filepath=fig_dir / "dsd_different_condevaps_selection_criteria")
plt.show()
condonly_selected_clusters
# %%
fig = dsdplots.plot_bottom_top_heavy_evaporation_dsds(microphysics_datasets_norm, condonly_selected_clusters, condonly_selected_clusters_colors)
save_figure(fig=fig, filepath=fig_dir / "dsd_different_condevaps")
plt.show()
# %%
fig = dsdplots.plot_figure_appdx_2_version2(microphysics_datasets_norm, extra_ds_normalized, condonly_selected_clusters, condonly_selected_clusters_colors)
save_figure(fig=fig, filepath=fig_dir_2 / "fig_appdx_2_version2")
plt.show()
# %%
fig = dsdplots.plot_figure_appdx_2(microphysics_datasets_norm, extra_ds_normalized, condonly_selected_clusters)
save_figure(fig=fig, filepath=fig_dir_2 / "fig_appdx_2")
plt.show()
# %%
fig = dsdplots.plot_figure_appdx_4(microphysics_datasets_norm, extra_ds_normalized, collisions_selected_clusters, microphysics_styles)
save_figure(fig=fig, filepath=fig_dir_2 / "fig_appdx_4")
plt.show()

# %%
