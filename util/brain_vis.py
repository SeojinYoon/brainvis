
# Common Libraries
import os, subprocess, glob
import numpy as np
import nibabel as nb
from pathlib import Path
from collections.abc import Iterable

# matplotlib
import matplotlib.pylab as plt
from matplotlib.colors import to_hex

# Vedo
from vedo import load, Text2D, Sphere, Plotter

# Custom Libraries
from brain_mask_util import untangle_mask_img
from afni_extension import cluster_infos

def cluster_to_mesh(cluster_map_path, 
                    cluster_numbers,
                    save_dir_path,
                    is_remove_cluster_map = True):
    """
    This function makes vtk files. Each vtk file represents one cluster.
    
    :param cluster_map_path: cluster path path(string) 
        -des: this map contains many clusters which are mostly made by the command - 3dClusterize prefmap.
    :param cluster_numbers: These values denote the order of cluster(list)
        -des: 
        -eType: int
        
        ex) [1,2,3]
    :param save_dir_path: Directory path to save vtk file(string) 

    return: vtk file paths(list - string)
    """
    cluster_numbers = list(cluster_numbers)
    
    # File validation check - cluster map
    if not os.path.exists(cluster_map_path) or len(cluster_numbers) == 0:
        return None
    
    # load cluster map
    cluster_map_img = nb.load(cluster_map_path)
    
    # Convert from cluster map to each cluster img
    each_cluster_imgs = untangle_mask_img(mask = cluster_map_img, 
                                          select_values = cluster_numbers)

    mesh_file_paths = []
    file_name = Path(cluster_map_path).stem
    for cluster_img, cluster_number in zip(each_cluster_imgs, cluster_numbers):
        # Paths
        ind_cluster_path = os.path.join(save_dir_path, f"{file_name}_cluster{str(cluster_number).zfill(3)}")
        nifti_file_path = ind_cluster_path + ".nii"
        mesh_file_path = ind_cluster_path + ".vtk"

        # Save each cluster img
        nb.save(cluster_img, ind_cluster_path)
        
        # Convert nii file to vtk file to make mesh
        command = f"nii2mesh {nifti_file_path} {mesh_file_path}"
        output = subprocess.check_output(command, shell=True)

        # Remove each cluster's nifti file
        os.system(f"rm {nifti_file_path}")

        # Stack mesh path
        mesh_file_paths.append(mesh_file_path)

    # Remove cluster map file
    if is_remove_cluster_map:
        os.system(f"rm {cluster_map_path}")

    return mesh_file_paths

def make_mesh_fromRM(roi_manager, 
                     search_keywords,
                     save_dir_path,
                     p = None,
                     s = None,
                     i = None,
                     r = None):
    """
    Make mesh(.vtk) files from Roi Manager
    
    :param roi_manager: roi_manager ex) fan_roi_manager
    :param search_keywords: search keywords(list - list)
    :param save_dir_path: Directory path for saving vtk files
    
    return vtk file paths
    """
    mesh_paths = []
    for keywords in search_keywords:
        file_name = "_".join(keywords)
        path = os.path.join(save_dir_path, file_name)
        nii_path = path + ".nii"
        vtk_path = path + ".vtk"

        if not os.path.exists(vtk_path):
            mask = roi_manager.search_roi(keywords)

            # save nifti
            nb.save(mask.mask_nifti_img, nii_path)

            # make mesh
            command_format = "nii2mesh {nifti_path} {options} {mesh_path}"
            
            options = []
            if p is not None:
                options.append(f"-p {p}")
            if s is not None:
                options.append(f"-s {s}")
            if i is not None:
                options.append(f"-i {i}")
            if r is not None:
                options.append(f"-r {r}") 
            
            options = " ".join(options)

            command = command_format.format(nifti_path = nii_path,
                                            mesh_path = vtk_path,
                                            options = options)
            print(command)
            output = subprocess.check_output(command, shell=True)
            
            os.system("rm " + nii_path)
            
            # stack result
            mesh_paths.append(vtk_path)
        else:
            mesh_paths.append(vtk_path)
            
    return mesh_paths

def make_cluster_spheres(cluster_df, color, loc_type = "center"):
    """
    Make sphere from cluster information
    
    :param cluster_df: (DataFrame)
    :param color: color(hex)
    :param loc_type: center, peak(string) - must match with cluseter df's atlas query method
    
    return (list - sphere)
    """
    if loc_type == "center":
        select_columns = ["CM LR", "CM PA", "CM IS", "name"]
    elif loc_type == "peak":
        select_columns = ["MI LR", "MI PA", "MI IS", "name"]
    
    if cluster_df is None:
        return None
    
    # Cluster location
    cluster_locs = cluster_df[select_columns]
    
    # Loop over all location
    spheres = []
    for row_i in range(len(cluster_locs)):
        cluster_data = cluster_locs.iloc[row_i]
        x, y, z = np.array(cluster_data[:-1]).astype(float).astype(int)
        name = cluster_data[-1]

        # Make sphere
        s = Sphere(pos = [x,y,z], r = 2, c = color).lighting('glossy')
        s.name = name

        spheres.append(s)
    return spheres
    
def make_cluster_meshes(cluster_df, mesh_paths, color):
    """
    Make mesh from cluster information.

    :param cluster_df: DataFrame with cluster information
    :param mesh_paths: mesh paths corresponding to cluster_df
    :param color: color (hex string)
    :return: list of vedo Mesh objects
    """
    if cluster_df is None or len(cluster_df) == 0:
        return []

    cluster_numbers = cluster_df.index + 1
    clusters = []

    for cluster_number in cluster_numbers:
        target_key = f"cluster{str(cluster_number).zfill(3)}"
        matching_paths = [p for p in mesh_paths if target_key in p]
        if not matching_paths:
            continue

        mesh_path = matching_paths[0]
        cluster_mesh = load(mesh_path).color(color)
        cluster_mesh.name = cluster_df.loc[cluster_number - 1]["name"]
        clusters.append(cluster_mesh)

    return clusters

def load_and_style_rois(roi_vtk_files, roi_style_info=None):
    """
    Load ROI VTK files and apply color, opacity, lighting, and geometric adjustments.
    
    :param roi_vtk_files: list of paths to ROI VTK files
    :param roi_style_info: dict with keys (colors, opacities, lightenings, line_widths, line_colors, adjust_methods)
    :return: list of styled vedo Mesh objects
    """
    if not roi_vtk_files:
        return []

    roi_style_info = roi_style_info or {}
    colors = roi_style_info.get("colors", "#929591")
    opacities = roi_style_info.get("opacities", 0.1)
    lightings = roi_style_info.get("lightenings", "glossy")
    line_widths = roi_style_info.get("line_widths", None)
    line_colors = roi_style_info.get("line_colors", None)
    adjust_methods = roi_style_info.get("adjust_methods", [])

    def _get_val(prop, idx, default=None):
        if prop is None:
            return default
        if isinstance(prop, Iterable) and not isinstance(prop, (str, bytes)):
            return prop[idx] if idx < len(prop) else default
        return prop

    roi_volumes = []
    for i, vtk_path in enumerate(roi_vtk_files):
        volume = load(vtk_path)

        # Style attributes
        volume.color(_get_val(colors, i, "#929591"))
        volume.opacity(_get_val(opacities, i, 0.1))
        volume.lighting(_get_val(lightings, i, "glossy"))

        lw = _get_val(line_widths, i, None)
        if lw is not None:
            volume.lw(lw)

        lc = _get_val(line_colors, i, None)
        if lc is not None:
            volume.lc(lc)

        # Mesh geometry adjustments
        for adjust_type, params in adjust_methods:
            if adjust_type == "subdivide":
                volume = volume.subdivide(method=params.get("method", 0), mel=params.get("mel", None))
            elif adjust_type == "smooth":
                volume = volume.smooth(niter=params.get("niter", 15), boundary=params.get("boundary", False))
            elif adjust_type == "clean":
                volume = volume.clean()
            elif adjust_type == "is_normal":
                volume = volume.computeNormals(
                    points=params.get("points", True),
                    cells=params.get("cells", True),
                    consistency=params.get("consistency", True)
                )
            elif adjust_type == "decimate":
                volume = volume.decimate(
                    fraction=params.get("fraction", 0.5),
                    method=params.get("method", "quadric"),
                    boundaries=params.get("boundaries", False)
                )

        volume.pickable(False)
        roi_volumes.append(volume)

    return roi_volumes


def build_clusters(
    stat_map_paths,
    temp_dir_path,
    cluster_plot_style="point",
    atlas_query_method="center",
    atlas_name="Haskins_Pediatric_Nonlinear_1.01",
    thresholds=None,
    cluster_size=40,
    NN_level=1,
    stat_indexes=None,
    cluster_map_colors=None,
):
    """
    Query cluster statistics from stat maps and construct visual actors (meshes or spheres).

    :param stat_map_paths: list of paths to statistical NIfTI maps (e.g., from 3dttest++)
    :param temp_dir_path: directory path to store temporary cluster masks and meshes
    :param cluster_plot_style: visualization style for clusters ('point' or 'mesh')
    :param atlas_query_method: coordinate method for atlas query ('center' or 'peak')
    :param atlas_name: reference atlas name for anatomical labeling
    :param thresholds: list of statistical thresholds for clustering
    :param cluster_size: minimum cluster volume in voxels
    :param NN_level: neighborhood connectivity level (1, 2, or 3)
    :param stat_indexes: sub-brick indices within stat maps
    :param cluster_map_colors: color(s) for cluster visualization
    :return: list of cluster actor groups (one group per stat map)
    """
    n_stat = len(stat_map_paths)
    if stat_indexes is None:
        stat_indexes = np.repeat(1, n_stat)

    # Prepare pref_maps for mesh mode
    if cluster_plot_style == "mesh":
        for old_mask in Path(temp_dir_path).glob("*_clust_mask.nii"):
            old_mask.unlink(missing_ok=True)

        pref_maps = [
            os.path.join(temp_dir_path, f"{Path(path).stem}_clust_mask.nii")
            for path in stat_map_paths
        ]
    else:
        pref_maps = None

    # Query cluster information via AFNI / atlas query
    cluster_dfs = cluster_infos(
        stat_map_paths=stat_map_paths,
        thresholds=thresholds,
        cluster_sizes=np.repeat(cluster_size, n_stat),
        is_positive=True,
        pref_maps=pref_maps,
        atlas_query_method=atlas_query_method,
        atlas_name=atlas_name,
        NN_level=NN_level,
        stat_indexes=stat_indexes,
    )

    # Generate cluster meshes if requested
    if cluster_plot_style == "mesh":
        cluster_mesh_paths = []
        for model_i, pref_path in enumerate(pref_maps):
            df = cluster_dfs[model_i]
            if df is None:
                cluster_mesh_paths.append([])
                continue
            cluster_numbers = df.index + 1
            paths = cluster_to_mesh(
                cluster_map_path=pref_path,
                cluster_numbers=cluster_numbers,
                save_dir_path=temp_dir_path,
            )
            cluster_mesh_paths.append(paths)

    # Determine cluster colors
    if cluster_map_colors is None:
        raw_colors = plt.cm.rainbow(np.linspace(0, 1, len(cluster_dfs)))[::-1]
        cluster_map_colors = [to_hex(c) for c in raw_colors]
    elif isinstance(cluster_map_colors, str):
        cluster_map_colors = np.repeat(cluster_map_colors, len(cluster_dfs))

    # Build 3D actors (spheres or meshes)
    clusters = []
    if cluster_plot_style == "point":
        for cluster_df, color in zip(cluster_dfs, cluster_map_colors):
            clusters.append(make_cluster_spheres(cluster_df=cluster_df, color=color))
    else:
        for cluster_df, mesh_paths, color in zip(cluster_dfs, cluster_mesh_paths, cluster_map_colors):
            clusters.append(make_cluster_meshes(cluster_df=cluster_df, mesh_paths=mesh_paths, color=color))

    for cluster_group in clusters:
        for c in cluster_group:
            c.opacity(1.0)

    return clusters


class BrainClusterVisualizer:
    """
    Interactive 3D Brain Cluster Visualizer based on vedo.
    Manages data loading, AFNI cluster extraction, ROI styling, and interactive 3D rendering.
    """

    def __init__(
        self,
        stat_map_paths,
        temp_dir_path,
        base_brain_nii_path=None,
        roi_vtk_files=None,
        roi_style_info=None,
    ):
        """
        Initialize the BrainClusterVisualizer.

        :param stat_map_paths: list of statistical NIfTI map paths
        :param temp_dir_path: directory path for caching intermediate mesh and mask files
        :param base_brain_nii_path: path to anatomical template / group mask NIfTI
        :param roi_vtk_files: list of ROI VTK file paths
        :param roi_style_info: dict configuring ROI styles (colors, opacities, line widths, etc.)
        """
        self.stat_map_paths = list(stat_map_paths)
        self.temp_dir_path = Path(temp_dir_path)
        self.temp_dir_path.mkdir(parents=True, exist_ok=True)
        self.base_brain_nii_path = base_brain_nii_path
        self.roi_vtk_files = list(roi_vtk_files) if roi_vtk_files else []
        self.roi_style_info = roi_style_info or {}

        # Default cluster analysis parameters
        self.cluster_params = {
            "thresholds": None,
            "cluster_size": 40,
            "NN_level": 1,
            "atlas_query_method": "center",
            "atlas_name": "Haskins_Pediatric_Nonlinear_1.01",
            "stat_indexes": None,
        }

        # Cached visual actors
        self._roi_volumes = None
        self._clusters = None
        self._cluster_plot_style = None
        self._cluster_map_colors = None

    def set_cluster_params(
        self,
        thresholds=None,
        cluster_size=40,
        NN_level=1,
        atlas_query_method="center",
        atlas_name="Haskins_Pediatric_Nonlinear_1.01",
        stat_indexes=None,
    ):
        """
        Configure cluster extraction and atlas query parameters.

        :param thresholds: list of statistical thresholds for clustering
        :param cluster_size: minimum cluster volume in voxels
        :param NN_level: neighborhood connectivity level (1, 2, or 3)
        :param atlas_query_method: coordinate method for atlas query ('center' or 'peak')
        :param atlas_name: reference atlas name for anatomical labeling
        :param stat_indexes: sub-brick indices within stat maps
        :return: self (for method chaining)
        """
        new_params = {
            "thresholds": thresholds,
            "cluster_size": cluster_size,
            "NN_level": NN_level,
            "atlas_query_method": atlas_query_method,
            "atlas_name": atlas_name,
            "stat_indexes": stat_indexes,
        }
        if new_params != self.cluster_params:
            self.cluster_params = new_params
            self._clusters = None  # Invalidate cached clusters
        return self

    def set_rois(self, roi_vtk_files, roi_style_info=None):
        """
        Update ROI VTK files and styling.

        :param roi_vtk_files: list of ROI VTK file paths
        :param roi_style_info: dict configuring ROI styles (colors, opacities, line widths, etc.)
        :return: self (for method chaining)
        """
        self.roi_vtk_files = list(roi_vtk_files)
        if roi_style_info is not None:
            self.roi_style_info = roi_style_info
        self._roi_volumes = None  # Invalidate cached ROIs
        return self

    def _prepare_base_brain(self):
        """Ensure base brain mesh is generated if template NIfTI is specified."""
        if self.base_brain_nii_path and os.path.exists(self.base_brain_nii_path):
            base_brain_vtk_path = self.temp_dir_path / f"{Path(self.base_brain_nii_path).stem}.vtk"
            if not base_brain_vtk_path.exists():
                subprocess.run(["nii2mesh", str(self.base_brain_nii_path), str(base_brain_vtk_path)], check=True)

    def get_roi_volumes(self):
        """Retrieve or build ROI mesh actors."""
        if self._roi_volumes is None:
            self._roi_volumes = load_and_style_rois(self.roi_vtk_files, self.roi_style_info)
        return self._roi_volumes

    def get_clusters(self, cluster_plot_style="point", cluster_map_colors=None):
        """
        Retrieve or build cluster actors (cached unless style/colors change).

        :param cluster_plot_style: visualization style for clusters ('point' or 'mesh')
        :param cluster_map_colors: color(s) for cluster visualization
        :return: list of cluster actor groups
        """
        if (
            self._clusters is None
            or self._cluster_plot_style != cluster_plot_style
            or self._cluster_map_colors != cluster_map_colors
        ):
            self._cluster_plot_style = cluster_plot_style
            self._cluster_map_colors = cluster_map_colors
            self._clusters = build_clusters(
                stat_map_paths=self.stat_map_paths,
                temp_dir_path=str(self.temp_dir_path),
                cluster_plot_style=cluster_plot_style,
                atlas_query_method=self.cluster_params["atlas_query_method"],
                atlas_name=self.cluster_params["atlas_name"],
                thresholds=self.cluster_params["thresholds"],
                cluster_size=self.cluster_params["cluster_size"],
                NN_level=self.cluster_params["NN_level"],
                stat_indexes=self.cluster_params["stat_indexes"],
                cluster_map_colors=cluster_map_colors,
            )
        return self._clusters

    def show(
        self,
        cluster_plot_style="point",
        cluster_map_colors=None,
        background_color="black",
    ):
        """
        Render the 3D scene with interactive slider and area inquiry.

        :param cluster_plot_style: visualization style for clusters ('point' or 'mesh')
        :param cluster_map_colors: color(s) for cluster visualization
        :param background_color: viewer background color (e.g. 'black', 'white', or hex)
        :return: self
        """
        self._prepare_base_brain()
        roi_volumes = self.get_roi_volumes()
        clusters = self.get_clusters(cluster_plot_style, cluster_map_colors)

        n_stat = len(self.stat_map_paths)
        file_names = [Path(p).stem for p in self.stat_map_paths]
        all_objs = list(roi_volumes)
        for cluster_group in clusters:
            all_objs.extend(cluster_group)

        msg = Text2D("", pos="bottom-center", c="k", bg="r9", alpha=0.8)
        plotter = Plotter(axes=1, bg=background_color)

        def on_mouse_click(evt):
            if not evt.actor:
                return
            try:
                actor_name = getattr(evt.actor, "name", "")
                if actor_name and actor_name != "Mesh":
                    msg.text(f"area name: {actor_name}")
            except Exception:
                pass

        plotter.add_callback("mouse click", on_mouse_click)

        slider_state = {"current": 0}
        x_indexes = np.arange(n_stat)

        def set_clusters_opacity(cluster_list, opacity):
            for c in cluster_list:
                c.opacity(opacity)

        def on_slider(widget, event):
            max_val = widget.GetSliderRepresentation().GetMaximumValue()
            t = widget.GetSliderRepresentation().GetCurrentT()
            current_range = int(np.round(t * max_val, 0))

            if slider_state["current"] == current_range:
                return
            slider_state["current"] = current_range

            if current_range == 0:
                msg.text("show All")
                for idx in x_indexes:
                    set_clusters_opacity(clusters[idx], 1.0)
            else:
                selected_idx = current_range - 1
                msg.text(f"show {file_names[selected_idx]}")
                for idx in x_indexes:
                    set_clusters_opacity(clusters[idx], 1.0 if idx == selected_idx else 0.0)

        plotter.add_slider(
            sliderfunc=on_slider,
            xmin=0,
            xmax=n_stat,
            value=0,
            c="blue",
            pos="bottom-right-vertical",
            title="stat_index",
        )

        plotter.show(all_objs, msg, __doc__, axes=2, zoom=1.2)
        plotter.close()
        return self

