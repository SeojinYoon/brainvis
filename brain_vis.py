
# Common Libraries
import os, subprocess, glob
import numpy as np
import nibabel as nb
from pathlib import Path
from collections.abc import Iterable

# matplotlib
from matplotlib import cm
import matplotlib.pylab as plt
from matplotlib.colors import to_hex
from skimage import io

# Vedo
from vedo import load, Text2D, Sphere, Plotter, Light, Point

# Custom Libraries
from brain_mask_util import untangle_mask_img
from string_util import search_stringAcrossTarget
from enum_util import File_validation
from afni_extension import cluster_infos
from higher_function_util import flatten

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
            if p != None:
                options += f"-p {p}"
            if s != None:
                options += f"-s {s}"
            if i != None:
                options += f"-i {i}"
            if r != None:
                options += f"-r {r}" 
            
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
    
    if type(cluster_df) == type(None):
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
    Make mesh from cluster information
    
    :param cluster_df: (DataFrame)
    :param mesh_paths: mesh paths corresponding to cluster_df
    :param color: color(hex)
    
    return (list - Mesh)
    """
    if type(cluster_df) == type(None) or len(cluster_df) == 0:
        return []
    
    cluster_numbers = cluster_df.index + 1

    clusters = []
    for cluster_number in cluster_numbers:
        mesh_path = search_stringAcrossTarget(targets = mesh_paths, 
                                              search_keys = [f"cluster{str(cluster_number).zfill(3)}"],
                                              validation_type = File_validation.exist_only)
        cluster_mesh = load(mesh_path).color(color)
        cluster_mesh.name = cluster_df.loc[cluster_number - 1]["name"]
        print(f"cluster{cluster_number}, name: {cluster_mesh.name}")
        clusters.append(cluster_mesh)
        
    return clusters

def base_roi_keywords():
    search_keywords = [
        ["Rt", "superior", "frontal", "gyrus"],
        ["Lt", "superior", "frontal", "gyrus"],
        ["Rt", "middle", "frontal", "gyrus"],
        ["Lt", "middle", "frontal", "gyrus"],
        ["Rt", "inferior", "frontal", "gyrus"],
        ["Lt", "inferior", "frontal", "gyrus"],
        ["Rt", "precentral"],
        ["Lt", "precentral"],
        ["Rt", "postcentral"],
        ["Lt", "postcentral"],
        ["Rt", "inferior", "parietal"],
        ["Lt", "inferior", "parietal"],
        ["Rt", "superior", "parietal"],
        ["Lt", "superior", "parietal"],
        ["Rt", "orbital"],
        ["Lt", "orbital"],
        ["Rt", "paracentral"],
        ["Lt", "paracentral"],
        ["Rt", "precuneus"],
        ["Lt", "precuneus"],
        ["Rt", "thalamus"],
        ["Lt", "thalamus"],
        ["Rt", "insular"],
        ["Lt", "insular"],
        ["Rt", "medioventral", "occipital"],
        ["Lt", "medioventral", "occipital"],
        ["Rt", "lateral", "occipital"],
        ["Lt", "lateral", "occipital"],
        ["Rt", "cerebellum"],
        ["Lt", "cerebellum"],
        ["Rt", "substantia", "nigra"],
        ["Lt", "substantia", "nigra"],
        ["Rt", "red", "nucleus"],
        ["Lt", "red", "nucleus"],
        ["Rt", "fusiform"],
        ["Lt", "fusiform"],
        ["Rt", "precuneus"],
        ["Lt", "precuneus"],
        ["Rt", "cingulate"],
        ["Lt", "cingulate"],
        ["Rt", "parahippocampal"],
        ["Lt", "parahippocampal"],
        ["Rt", "amygdala"],
        ["Lt", "amygdala"],
        ["Rt", "hippocampus"],
        ["Lt", "hippocampus"],
        ["Rt", "basal", "ganglia"],
        ["Lt", "basal", "ganglia"],
        ["Hypothalamus"],
        ["Rt" , "substantia", "nigra"],
        ["Lt" , "substantia", "nigra"],
        ["Brainstem"],
    ]

    return search_keywords

def show_clusterize_brain(
    stat_map_paths,
    cluster_dir_path,
    base_brain_nii_path,
    roi_vtk_files,
    cluster_plot_style = "point", # mesh, point
    atlas_query_method = "center", # peak, center
    atlas_name = "Haskins_Pediatric_Nonlinear_1.01",
    thresholds = None,
    cluster_size = 40,
    NN_level = 1,
    cluster_map_colors = None,
    stat_indexes = None,
    background_color = "black",
    roi_style_info = {},
    is_custom_lightening = False,
    lightening_style_info = {},
    axis_info = {},
    ):
    """
    Show clusterize brain

    :param stat_map_paths: path of stat map(list - string)
        -des: The path is made by the command - 3dttest++.
        -example: [ "/clmn/users/ttest.nii" ]
    :param cluster_dir_path: path to save files created by clustering analysis
        -example: "/Users/clmn/Downloads/vedo_vis/brain_rois/clusters"
    :param base_brain_nii_path: basic brain image(string)
        -example: "/Users/clmn/Downloads/vedo_dir/group_mask.nii"
    :param roi_vtk_files: paths of roi vtk file
        -example: ["/Users/clmn/Downloads/vedo_vis/brain_rois/a.vtk"]
     :param n_cluster_criterias: Each statmap needs to have under the number of cluster.(Int)
        -example: [5]
    :param cluster_plot_style: plotting style of cluster(string)
        -kind: mesh, point
    :param atlas_query_method: Find atlas based on the location
        -kind: peak, center
    :param atlas_name: atlas name
        -kind: "Haskins_Pediatric_Nonlinear_1.01"
    :param thresholds: Thresholds to do clustering analysis [list - float]
        -example: [3.14]
    :param cluster_size: The cluster's voxel must exceed the size.(Int)
        -example: 40
    :param NN_level: NN level(int)
        -example: 1
    :param roi_style_info.get: style of roi(dictionary)
        -example: colors : [ "#cdcd00" ], opacities : [ 0.3 ], lightenings : [ "glossy" ], line_widths : [ 1 ], line_colors : [ "#000000" ], adjust_methods : [("smooth", {}), ("decimate", { "fraction" : 0.9})]
    :param cluster_map_colors: colors to visualize cluster map(list - string)
        -example: [ "#cdcd00" ]
    :param stat_indexes: statmap index of file_paths(list - int)
        -example: [1]
    :param background_color: background color(string)
        -example: "#000000"
    :param is_custom_lightening: is_custom_lightening (boolean)
        -example: False
    :param axis_info: axis information(dictionary)
        -example: is_mirror_x : True, is_mirror_y : True, is_mirror_z : True

    """
    n_stat = len(stat_map_paths)
    file_names = [Path(path).stem for path in stat_map_paths]

    if type(stat_indexes) == type(None):
        stat_indexes = np.repeat(1, n_stat)
    
    # Load base brain
    base_brain_vtk_path = os.path.join(cluster_dir_path, f"{Path(base_brain_nii_path).stem}.vtk")
    os.system(f"nii2mesh {base_brain_nii_path} {base_brain_vtk_path}")
    print("base brain vtk path: ", base_brain_vtk_path)

    base_brain_vtk_volume = load(base_brain_vtk_path).opacity(0.1)

    if cluster_plot_style == "mesh":
        # Remove files if cluster mask exists
        clust_mask_paths = glob.glob(os.path.join(cluster_dir_path, "*_clust_mask.nii"))
        for path in clust_mask_paths:
            os.system(f"rm {path}")
        
        # File path of cluster mask
        pref_maps = []
        for file_path in stat_map_paths:
            path = os.path.join(cluster_dir_path, file_path.split(os.sep)[-1].split(".")[0] + "_clust_mask.nii")
            pref_maps.append(path)
    else:
        pref_maps = None

    # query based on spm coordinate
    cluster_dfs = cluster_infos(stat_map_paths = stat_map_paths,
                                thresholds = thresholds,
                                cluster_sizes = np.repeat(cluster_size, n_stat),
                                is_positive = True,
                                pref_maps = pref_maps,
                                atlas_query_method = atlas_query_method,
				atlas_name = atlas_name,
                                NN_level = NN_level,
                                stat_indexes = stat_indexes)

    # Make cluster vtk files
    if cluster_plot_style == "mesh":
        cluster_mesh_paths = []
        for model_i, path in enumerate(pref_maps):
            if type(cluster_dfs[model_i]) == type(None):
                cluster_mesh_paths.append([])
                continue

            cluster_numbers = cluster_dfs[model_i].index + 1
            paths = cluster_to_mesh(cluster_map_path = pref_maps[model_i], 
                                    cluster_numbers = cluster_numbers, 
                                    save_dir_path = cluster_dir_path)
            cluster_mesh_paths.append(paths)

    # ROI volume
    roi_colors = roi_style_info.get("colors", "#929591")
    roi_opacities = roi_style_info.get("opacities", 0.1)
    roi_lightenings = roi_style_info.get("lightenings", "glossy")
    roi_line_widths = roi_style_info.get("line_widths", None)
    roi_line_colors = roi_style_info.get("line_colors", None)

    roi_vtk_volumes = []
    for i, vtk_path in enumerate(roi_vtk_files):
        roi_vtk_volume = load(vtk_path)

	    # color
        if (isinstance(roi_colors, Iterable)) and type(roi_colors) != str:
            roi_vtk_volume = roi_vtk_volume.color(roi_colors[i])
        else:
            roi_vtk_volume = roi_vtk_volume.color(roi_colors)
	
	    # opacity
        if isinstance(roi_opacities, Iterable):
            roi_vtk_volume = roi_vtk_volume.opacity(roi_opacities[i])
        else:
            roi_vtk_volume = roi_vtk_volume.opacity(roi_opacities)
	
	    # lightening
        if type(roi_lightenings) != str and isinstance(roi_lightenings, Iterable):
            roi_vtk_volume = roi_vtk_volume.lighting(roi_lightenings[i])
        else:
            roi_vtk_volume = roi_vtk_volume.lighting(roi_lightenings)
	
	    # line width
        if type(roi_line_widths) != type(None) and isinstance(roi_line_widths, Iterable):
            roi_vtk_volume = roi_vtk_volume.lw(roi_line_widths[i])
        elif type(roi_line_widths) != type(None) and not isinstance(roi_line_widths, Iterable):
            roi_vtk_volume = roi_vtk_volume.lw(roi_line_widths)
            
	    # line color
        if type(roi_line_colors) != type(None) and type(roi_line_colors) != str and isinstance(roi_line_colors, Iterable):
            roi_vtk_volume = roi_vtk_volume.lc(roi_line_colors[i])
        elif type(roi_line_colors) == str:
            roi_vtk_volume = roi_vtk_volume.lc(roi_line_colors)


        roi_adjusts = roi_style_info.get("adjust_methods", [])
        for adjust_type, value in roi_adjusts:
            if adjust_type == "subdivide":
                method = value.get("method", 0)
                mel = value.get("mel", None)

                roi_vtk_volume = roi_vtk_volume.subdivide(method = method, mel = mel)
            if adjust_type == "smooth":
                niter = value.get("niter", 15)
                boundary = value.get("boundary", False)

                roi_vtk_volume = roi_vtk_volume.smooth(niter = niter, boundary = boundary)
            if adjust_type == "clean":
                roi_vtk_volume = roi_vtk_volume.clean()
            if adjust_type == "is_normal":
                points = value.get("points", True)
                cells = value.get("cells", True)
                consistency = value.get("consistency", True)

                roi_vtk_volume = roi_vtk_volume.computeNormals(points = points, cells = cells, consistency = consistency)
            if adjust_type == "decimate":
                fraction = value.get("fraction", 0.5)
                method = value.get("method", "quadric")
                boundaries = value.get("boundaries", False)

                roi_vtk_volume = roi_vtk_volume.decimate(fraction = fraction, method = method, boundaries = boundaries)

        # Set enable interaction as false
        roi_vtk_volume.pickable(False)

        # Accumulate element
        roi_vtk_volumes.append(roi_vtk_volume)

    # cluster color
    if cluster_map_colors == None:
        cluster_map_colors = plt.cm.rainbow(np.linspace(0, 1, len(cluster_dfs)))
        cluster_map_colors = cluster_map_colors[::-1]
        io.imshow(np.expand_dims(cluster_map_colors[:,:-1], 1))
        cluster_map_colors = [to_hex(color) for color in cluster_map_colors]
    elif type(cluster_map_colors) == str:
        cluster_map_colors = np.repeat(cluster_map_colors, len(cluster_dfs))

    # stack result
    clusters = []

    if cluster_plot_style == "point":
        for cluster_df, color in zip(cluster_dfs, cluster_map_colors):
            clusters_ = make_cluster_spheres(cluster_df = cluster_df, color = color)
            clusters.append(clusters_)
    else:
        for cluster_df, mesh_paths, color in zip(cluster_dfs, cluster_mesh_paths, cluster_map_colors):
            clusters_ = make_cluster_meshes(cluster_df = cluster_df, mesh_paths = mesh_paths, color = color)
            clusters.append(clusters_)

    for c in flatten(clusters):
        c.opacity(1.0)

    # base brain opacity
    base_brain_volume = base_brain_vtk_volume.opacity(0)

    # control info
    control_var = {
        "mode" : "normal",
        "target" : None
    }

    # Interactive function
    slider_info = {
        "current" : 0
    }

    xmin = 0
    xmax = n_stat

    def mouse_click(evt):
        mouse_click_lightening(evt)

        if control_var["mode"] == "normal":
            if not evt.actor:
                return
            try:
                origin = evt.actor.c()
                # sil = evt.actor.c('red5')
                if evt.actor.name != "Mesh":
                    msg.text("area name: "+ evt.actor.name)
                else:
                    print(1)
                # plotter.remove('silu').add(sil)
            except:
                pass
        
        
        
    def show_clusters(clusters, opacity):
        for c in clusters:
            c.opacity(opacity)

    def slider(widget, event):    
        maximum = widget.GetSliderRepresentation().GetMaximumValue()
        v = widget.GetSliderRepresentation().GetCurrentT()
        
        current_range = int(np.round((v * maximum), 0))
        
        if slider_info["current"] == current_range:
            pass
        else:
            slider_info["current"] = current_range
            
            if current_range == 0:
                msg.text("show All")
                
                # All clusters
                for x_i in x_indexes:
                    show_clusters(clusters = clusters[x_i], opacity = 1.0)
            else:
                # if 1, show 0 index stat cluster
                x_i = current_range - 1
                
                msg.text(f"show {file_names[x_i]}")

                other_indexes = x_indexes[x_indexes != x_i]
                for i in other_indexes:
                    show_clusters(clusters = clusters[i], opacity = 0.0)
                    
                show_clusters(clusters = clusters[x_i], opacity = 1.0)

    """
    Lightening code start:
    """
    if is_custom_lightening:
        lightening_sphere_radius = lightening_style_info.get("radius", 15)
        lightening_sphere_color = lightening_style_info.get("sphere_color", "red")

        lightening_color = lightening_style_info.get("lightening_color", "w")

        # Make first lightening
        first_light_pos = [0,0,0]
        s1 = Point(r = lightening_sphere_radius).pos(first_light_pos).c(lightening_sphere_color)
        s1.name = "1"

        l1 = Light(first_light_pos, c = lightening_color, intensity = 1)
        l1.name = "Light1"

        # lightening info
        mapping_light_info = {
            s1.name : l1
        }

    def mouse_click_lightening(evt):
        if is_custom_lightening:
            if control_var["mode"] == "normal":
                pass
            elif control_var["mode"] == "select_light":
                if evt.actor != None:
                    control_var["target"] = evt.actor
                    auxilary_msg.text("name: " + control_var["target"].name)
                    plotter.render()
            elif control_var["mode"] == "move_light":
                if mapping_light_info.get(control_var["target"].name, None) != None:
                    point = control_var["target"]
                    lightening = mapping_light_info[point.name]
                    
                    mouse_pos_3d = plotter.computeWorldPosition(evt.picked2d)
                    point.SetPosition(mouse_pos_3d)
                    lightening.SetPosition(mouse_pos_3d)
                
                    plotter.render()
            elif control_var["mode"] == "make_light":
                point_number = int(sorted(list(mapping_light_info.keys()))[-1]) + 1
                
                mouse_pos_3d = plotter.computeWorldPosition(evt.picked2d)
                p = Point(r = lightening_sphere_radius).pos(mouse_pos_3d).c(lightening_sphere_color)
                p.name = str(point_number)

                l = Light(mouse_pos_3d, c = lightening_color, intensity = 1)
                l.name = f"Light{point_number}"
                
                mapping_light_info[p.name] = l
                
                plotter.add(p, l)
                plotter.render()

    def key_pressed(evt):
        if is_custom_lightening:
            # change mode
            mode = control_var["mode"]
            # normal -> select_light -> move_light -> make_light
            
            if evt["keyPressed"] == "backslash":
                if mode == "normal":
                    control_var["mode"] = "select_light"
                elif mode == "select_light":
                    control_var["mode"] = "move_light"
                elif mode == "move_light":
                    control_var["mode"] = "make_light"
                else:
                    control_var["mode"] = "normal"
                
                auxilary_msg.text("mode: " + control_var["mode"])
                
                plotter.render()
    """
    Lightening code end:
    """
    
    # msg
    msg = Text2D("", pos="bottom-center", c='k', bg='r9', alpha=0.8)
    auxilary_msg = Text2D("", pos="top-left", c='k', bg='r9', alpha=0.8)

    # plot
    plotter = Plotter(axes = 1, bg = background_color)
    plotter.add_callback('mouse click', mouse_click)
    plotter.add_callback("KeyPress", key_pressed)

    x_indexes = np.arange(n_stat)
    plotter.add_slider(
        sliderfunc = slider,
        xmin = xmin,
        xmax = xmax,
        value = 0,
        c = "blue",
        pos = "bottom-right-vertical",
        title = "stat_index"
    )

    # Show - What is set later appears first
    objs = roi_vtk_volumes + clusters
    objs = flatten(objs)

    # Change axis
    ch_objs = []
    for obj in objs:
        for axis_key in axis_info:
            if axis_info[axis_key] == False:
                continue

            if axis_key == "is_mirror_x":
                obj = obj.clone().mirror("x")
            if axis_key == "is_mirror_y":
                obj = obj.clone().mirror("y")
            if axis_key == "is_mirror_z":
                obj = obj.clone().mirror("z")

        ch_objs.append(obj)
    objs = ch_objs

    if is_custom_lightening:
        plotter.show(objs, msg, auxilary_msg, l1, s1,
                __doc__, 
                axes = 2,
                zoom=1.2)
    else:
        plotter.show(objs, msg, auxilary_msg,
                __doc__, 
                axes = 2,
                zoom=1.2)
    plotter.close()
