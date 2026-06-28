# Independent of representations
import sympy as sp
from itertools import combinations, product
from utils import Representation, Interval

#  --- Nodes translations ---
def moveright(node):
    return (node[0], node[1] + 1)

def moveleft(node):
    return (node[0], node[1] - 1)

def movedown(node):
    return (node[0] - 1, node[1])

def moveup(node):
    return (node[0] + 1, node[1])

def moveleftdown(node):
    return (node[0] - 1, node[1] - 1)

# --- Convert dimvec to lists ---
def dimvec_visual_grid(grid_str):
    # 1. Strip leading/trailing whitespace and split into rows.
    # splitlines() handles newline characters automatically.
    lines = grid_str.strip().splitlines()
    total_rows = len(lines)
    support_list = []

    # 2. Iterate over each row string.
    for i, line in enumerate(lines):
        # --- Coordinate conversion ---
        # Row i in the string (counted top to bottom, i = 0, 1, ...)
        # corresponds to actual coordinate r = max_row, ..., 0.
        r = total_rows - 1 - i
        
        # 3. Clean data: accept "0 1 0" (with spaces) or "010" (compact).
        # Remove all spaces.
        clean_line = line.replace(" ", "")
        
        # 4. Iterate over columns.
        for c, char in enumerate(clean_line):
            if char == '1':
                support_list.append((r, c))
    
    # Sort for readability, by row and then by column.
    support_list.sort(key=lambda x: (x[0], x[1]))
    
    return support_list, [total_rows, len(lines[0].replace(" ", ""))]


# --- Convex hull of an interval (independent of representations)---
def generate_coordinates_Rfree(gridsize, current_dim=0, current_coords=None):
    gsize = gridsize
    if current_coords is None:
        current_coords = []
    if current_dim == len(gsize):
        return [tuple(current_coords)]
    coordinates = []
    for i in range(gsize[current_dim]):
        new_coords = current_coords + [i]
        coordinates.extend(generate_coordinates_Rfree(gsize, current_dim + 1, new_coords))
    return coordinates

def generate_nodes_Rfree(gridsize):
    nodes = []
    gsize = gridsize
    for coordinates in generate_coordinates_Rfree(gsize):
        nodes.append(tuple(coordinates))
    return nodes

def is_smaller_Rfree(node1, node2):
    # 1. Check that coordinate dimensions match; raise an error otherwise.
    if len(node1) != len(node2):
        raise ValueError(f"Dimension mismatch: node1 has length {len(node1)}, but node2 has length {len(node2)}.")
    
    # 2. Compare coordinate-wise.
    return all(n1 <= n2 for n1, n2 in zip(node1, node2))

def convex_square_Rfree(node1, node2, gridsize):
    gsize = gridsize
    if not is_smaller_Rfree(node1, node2):
        return []
    if len(node1) != len(gsize) or len(node2) != len(gsize):
        raise ValueError(
            f"Dimension mismatch: nodes have lengths {len(node1)}, {len(node2)}, "
            f"but gridsize has length {len(gsize)}."
        )
    coordinate_ranges = [range(node1[i], node2[i] + 1) for i in range(len(gsize))]
    return [tuple(coordinates) for coordinates in product(*coordinate_ranges)]

def int_hull_Rfree(interval, gridsize):
    points = []
    gsize = gridsize
    if len(gsize) != 2:
        for src in interval.src:
            for snk in interval.snk:
                points += convex_square_Rfree(src, snk, gsize)
        points = list(set(points))
        for pt in points[:]:
            if to_be_removed_Rfree(pt, interval):
                points.remove(pt)
        return points

    for x_coord in range(gsize[0]):
        lower_y = None
        for src in interval.src:
            if src[0] <= x_coord:
                lower_y = src[1] if lower_y is None else min(lower_y, src[1])

        upper_y = None
        for snk in interval.snk:
            if x_coord <= snk[0]:
                upper_y = snk[1] if upper_y is None else max(upper_y, snk[1])

        if lower_y is None or upper_y is None or lower_y > upper_y:
            continue
        points.extend((x_coord, y_coord) for y_coord in range(lower_y, upper_y + 1))
    return points

def to_be_removed_Rfree(point, interval):
    for src in interval.src:
        if is_smaller_Rfree(point, src) and point != src:
            return True
    for snk in interval.snk:
        if is_smaller_Rfree(snk, point) and point != snk:
            return True
    return False

# --- Sources and sinks ---
def add_source_Rfree(pt, points):
    for other in points:
        if is_smaller_Rfree(other, pt) and other != pt:
            return False
    return True

def get_src_snk_Rfree(points):
    src, snk = [], []
    for pt in points:
        if add_source_Rfree(pt, points):
            src.append(pt)
        if add_sink_Rfree(pt, points):
            snk.append(pt)
    return src, snk

def add_sink_Rfree(pt, points):
    for other in points:
        if is_smaller_Rfree(pt, other) and other != pt:
            return False
    return True

# --- Find proper sources and sinks ---
def get_prop_src_snk_Rfree(points, gridsize):
    '''
    For each interval I, compute the source (resp. sink) of the proper up-set (down-set) of I 
    '''
    src_temp, snk_temp =  get_src_snk_Rfree(points)
    return get_prop_src_snk_from_extrema_Rfree(src_temp, snk_temp, gridsize)

def get_prop_src_snk_from_extrema_Rfree(src, snk, gridsize):
    '''
    Compute src(U(src) \\ I) and snk(D(snk) \\ I) directly from the source and
    sink antichains of a 2D interval I = [src, snk].
    '''
    gsize = gridsize
    if len(gsize) != 2:
        min_node = tuple(0 for _ in gsize)
        max_node = tuple(dim - 1 for dim in gsize)
        interval = Interval(src, snk)
        points = int_hull_Rfree(interval, gsize)
        upset = Interval(src, [max_node])
        downset = Interval([min_node], snk)
        prop_up = [x for x in int_hull_Rfree(upset, gsize) if x not in points]
        prop_down = [x for x in int_hull_Rfree(downset, gsize) if x not in points]
        prp_src = get_src_snk_Rfree(prop_up)[0]
        prp_snk = get_src_snk_Rfree(prop_down)[1]
        return prp_src, prp_snk

    x_bound, y_bound = gsize
    src_uuset = []
    best_up_y = y_bound
    for x_coord in range(x_bound):
        lower_y = None
        for source in src:
            if source[0] <= x_coord:
                lower_y = source[1] if lower_y is None else min(lower_y, source[1])
        if lower_y is None:
            continue

        upper_y = None
        for sink in snk:
            if x_coord <= sink[0]:
                upper_y = sink[1] if upper_y is None else max(upper_y, sink[1])

        if upper_y is None or lower_y > upper_y:
            threshold = lower_y
        else:
            threshold = upper_y + 1
        if threshold < y_bound and threshold < best_up_y:
            src_uuset.append((x_coord, threshold))
            best_up_y = threshold

    snk_ddset = []
    best_down_y = -1
    for x_coord in range(x_bound - 1, -1, -1):
        upper_y = None
        for sink in snk:
            if x_coord <= sink[0]:
                upper_y = sink[1] if upper_y is None else max(upper_y, sink[1])
        if upper_y is None:
            continue

        lower_y = None
        for source in src:
            if source[0] <= x_coord:
                lower_y = source[1] if lower_y is None else min(lower_y, source[1])

        if lower_y is None or lower_y > upper_y:
            threshold = upper_y
        else:
            threshold = lower_y - 1
        if threshold >= 0 and threshold > best_down_y:
            snk_ddset.append((x_coord, threshold))
            best_down_y = threshold

    return src_uuset, sorted(snk_ddset)

def get_prop_src_Rfree(points, gridsize):
    '''
    For each interval I, compute the source of the proper up-set of I 
    '''
    src_temp, snk_temp = get_src_snk_Rfree(points)
    prp_src = get_prop_src_snk_from_extrema_Rfree(src_temp, snk_temp, gridsize)[0]
    return prp_src

def get_prop_snk_Rfree(points, gridsize):
    '''
    For each interval I, compute the sink of the proper down-set of I 
    '''
    src_temp, snk_temp = get_src_snk_Rfree(points)
    prp_snk = get_prop_src_snk_from_extrema_Rfree(src_temp, snk_temp, gridsize)[1]
    return prp_snk

# --- Determine whether an element is in given interval ---
def is_in_interval_Rfree(node, interval):
    '''
    Check membership logically without generating the hull.
    '''
    # 1. Check whether the node lies in the convex hull range
    #    using the same logic as convex_square.
    # It must be greater than or equal to some src and less than or equal to some snk.
    larger_than_src = any(is_smaller_Rfree(src, node) for src in interval.src)
    smaller_than_snk = any(is_smaller_Rfree(node, snk) for snk in interval.snk)
    if not (larger_than_src and smaller_than_snk):
        return False

    # 2. Check whether the node should be removed,
    #    using the same logic as to_be_removed.
    # It must not be strictly below any src:
    # if node < src and node != src, it is outside the interval.
    for src in interval.src:
        if is_smaller_Rfree(node, src) and node != src:
            return False

    # It must not be strictly above any snk.
    for snk in interval.snk:
        if is_smaller_Rfree(snk, node) and node != snk:
            return False

    return True
    
# --- Compute the dimension vector of an interval representation ---
def int_dimvec_Rfree(points, gridsize):
    rows, cols = gridsize
    dimvec = sp.zeros(rows, cols)
    for r, c in points:
        if 0 <= r < rows and 0 <= c < cols:
            dimvec[r, c] = 1

    for i in range(rows - 1, -1, -1):
        array_str = "".join(str(dimvec[i, j]) for j in range(cols))
        print(array_str)

# --- List rays of fixed nodes ---
def get_horizontal_right_ray_nodes_Rfree(node, bound):
    """
    Return a list of nodes having the same 1st coordinate as 'node',
    but with 2nd coordinate >= node[1], up to the dimension limit 'bound'.
    """
    x, start_y = node
    # Generate nodes: (x, y) for y in [start_y, bound - 1]
    return [(x, y) for y in range(start_y, bound)]

def get_horizontal_left_ray_nodes_Rfree(node):
    """
    Return a list of nodes having the same 1st coordinate as 'node',
    but with 2nd coordinate <= node[1], down to 0.
    """
    x, start_y = node
    return [(x, y) for y in range(0, start_y + 1)]

def get_vertical_up_ray_nodes_Rfree(node, bound):
    """
    Return a list of nodes having the same 2nd coordinate as 'node',
    but with 1st coordinate >= node[0], up to the dimension limit 'bound'.
    """
    start_x, y = node
    # Generate nodes: (x, y) for x in [start_x, bound - 1]
    return [(x, y) for x in range(start_x, bound)]

def get_vertical_down_ray_nodes_Rfree(node):
    """
    Return a list of nodes having the same 2nd coordinate as 'node',
    but with 1st coordinate <= node[0], up to the dimension limit.
    """
    start_x, y = node
    return [(x, y) for x in range(0, start_x + 1)]

def out_of_grid_Rfree(node, gridsize):
    dimensions_grid = gridsize
    if node[0] not in range(0, dimensions_grid[0]) or node[1] not in range(0, dimensions_grid[1]):
        return True
    else:
        return False

# --- Radical approximations ---
def right_rad_apprx_inj_Rfree(interval, gridsize):
    '''
    Find right radical approximations of a given interval where morphisms are injective
    '''
    gsize = gridsize
    points = int_hull_Rfree(interval, gsize)
    prp_snk = get_prop_snk_Rfree(points, gsize)
    x_bound = gsize[0]
    y_bound = gsize[1]
    right_rad_apprx_inj = []

    for node in prp_snk:
        cant_move_up = is_in_interval_Rfree(moveup(node), interval)
        cant_move_right = is_in_interval_Rfree(moveright(node), interval)
        
        if cant_move_up and cant_move_right:
            new_interval_hull = points + [node]
            right_rad_apprx_inj.append(new_interval_hull)

        elif cant_move_up and not cant_move_right:
            new_interval_hull = points + get_horizontal_right_ray_nodes_Rfree(node, y_bound)
            right_rad_apprx_inj.append(new_interval_hull)

        elif cant_move_right and not cant_move_up:
            new_interval_hull = points + get_vertical_up_ray_nodes_Rfree(node, x_bound)
            right_rad_apprx_inj.append(new_interval_hull)
            
    return right_rad_apprx_inj

def right_rad_apprx_surj_Rfree(interval, gridsize):
    '''
    Find right radical approximations of a given interval where morphisms are surjective
    '''
    gsize = gridsize
    points = int_hull_Rfree(interval, gsize)
    points_set = set(points)
    snk = interval.snk
    src = interval.src
    x_bound = gsize[0]
    y_bound = gsize[1]
    right_rad_apprx_surj = []

    for node in snk:
        x_coord = node[0]
        y_coord = node[1]
        exists_in_int_leftdown = is_in_interval_Rfree(moveleftdown(node), interval) or out_of_grid_Rfree(moveleftdown(node), gsize)
        exists_in_int_left = is_in_interval_Rfree(moveleft(node), interval) or out_of_grid_Rfree(moveleft(node), gsize)
        exists_in_int_down = is_in_interval_Rfree(movedown(node), interval) or out_of_grid_Rfree(movedown(node), gsize)

        if not exists_in_int_leftdown and not exists_in_int_left and not exists_in_int_down:
            pass
        elif not exists_in_int_leftdown and not exists_in_int_left and exists_in_int_down:
            pass
        elif not exists_in_int_leftdown and not exists_in_int_down and exists_in_int_left:
            pass
        elif not exists_in_int_leftdown and exists_in_int_down and exists_in_int_left:
            should_remove_vertical = (is_in_interval_Rfree((0, y_coord), interval) and not is_in_interval_Rfree((0, y_coord + 1), interval))
            should_remove_horizontal = (is_in_interval_Rfree((x_coord, 0), interval) and not is_in_interval_Rfree((x_coord + 1, 0), interval))

            if should_remove_vertical:
                removal_nodes_v = get_vertical_down_ray_nodes_Rfree(node)
                new_interval_hull_v = list(points_set - set(removal_nodes_v))
                right_rad_apprx_surj.append(new_interval_hull_v)

            if should_remove_horizontal:
                removal_nodes_h = get_horizontal_left_ray_nodes_Rfree(node)
                new_interval_hull_h = list(points_set - set(removal_nodes_h))
                right_rad_apprx_surj.append(new_interval_hull_h) 
        else:
            new_interval_hull = list(points_set - set([node]))
            if len(new_interval_hull) != 0:
                right_rad_apprx_surj.append(new_interval_hull)
            else:
                pass

    return right_rad_apprx_surj
