import sympy as sp
from itertools import combinations

# --- Helper functions ---
def powerset(elements):
    """Generate all possible subsets of a set of elements."""
    all_subsets = []
    for subset_size in range(len(elements) + 1):
        all_subsets.extend(combinations(elements, subset_size))
    return [list(subset) for subset in all_subsets]

def flatten(xss):
    """Flatten a list of lists."""
    return [x for xs in xss for x in xs]

# #  --- Nodes translations ---
# def moveright(node):
#     return (node[0], node[1] + 1)

# def moveleft(node):
#     return (node[0], node[1] - 1)

# def movedown(node):
#     return (node[0] - 1, node[1])

# def moveup(node):
#     return (node[0] + 1, node[1])

# def moveleftdown(node):
#     return (node[0] - 1, node[1] - 1)

# # --- Convert dimvec to lists ---
# def dimvec_visual_grid(grid_str):
#     # 1. Strip leading/trailing whitespace and split by line.
#     # splitlines() handles newline characters automatically.
#     lines = grid_str.strip().splitlines()
#     total_rows = len(lines)
#     support_list = []

#     # 2. Iterate over each row string.
#     for i, line in enumerate(lines):
#         # --- Key coordinate conversion ---
#         # The i-th string row is counted from top to bottom, i=0, 1, ...
#         # The corresponding actual coordinate r should count downward from the maximum row to 0.
#         r = total_rows - 1 - i
        
#         # 3. Clean the data: allow "0 1 0" with spaces or compact "010" input.
#         # Remove all spaces directly.
#         clean_line = line.replace(" ", "")
        
#         # 4. Iterate over columns.
#         for c, char in enumerate(clean_line):
#             if char == '1':
#                 support_list.append((r, c))
    
#     # Sort the result for readability, first by row and then by column.
#     support_list.sort(key=lambda x: (x[0], x[1]))
    
#     return support_list, [total_rows, len(lines[0].replace(" ", ""))]


# # --- Convex hull of an interval (independent of representations)---
# def generate_nodes_Rfree(gridsize):
#     nodes = []
#     gsize = gridsize
#     for coordinates in generate_coordinates_Rfree(gsize):
#         nodes.append(tuple(coordinates))
#     return nodes

# def generate_coordinates_Rfree(gridsize, current_dim=0, current_coords=None):
#     gsize = gridsize
#     if current_coords is None:
#         current_coords = []
#     if current_dim == len(gsize):
#         return [tuple(current_coords)]
#     coordinates = []
#     for i in range(gsize[current_dim]):
#         new_coords = current_coords + [i]
#         coordinates.extend(generate_coordinates_Rfree(gsize, current_dim + 1, new_coords))
#     return coordinates

# def is_smaller_Rfree(node1, node2, gridsize):
#     return all(node1[i] <= node2[i] for i in range(len(gridsize)))

# def convex_square_Rfree(node1, node2, gridsize):
#     gsize = gridsize
#     if not is_smaller_Rfree(node1, node2, gsize):
#         return []
#     points = []
#     for coordinates in generate_nodes_Rfree(gsize):
#         if all(node1[i] <= coordinates[i] <= node2[i] for i in range(len(gsize))):
#             points.append(tuple(coordinates))
#     return points

# def int_hull_Rfree(interval, gridsize):
#     points = []
#     gsize = gridsize
#     for src in interval.src:
#         for snk in interval.snk:
#             points += convex_square_Rfree(src, snk, gsize)
#     points = list(set(points))
#     for pt in points[:]:
#         if to_be_removed_Rfree(pt, interval, gsize):
#             points.remove(pt)
#     return points

# def to_be_removed_Rfree(point, interval, gridsize):
#     for src in interval.src:
#         if is_smaller_Rfree(point, src, gridsize) and point != src:
#             return True
#     for snk in interval.snk:
#         if is_smaller_Rfree(snk, point, gridsize) and point != snk:
#             return True
#     return False


# --- Interval Class ---
class Interval:
    """Define intervals by their sources and sinks."""
    def __init__(self, src, snk):
        self.src = [tuple(s) for s in src]
        self.snk = [tuple(s) for s in snk]

# --- Representation Class ---
class Representation:
    """Quiver representation with symbolic matrices using SymPy."""
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.nodes = self.generate_nodes()
        self.edges = self.generate_edges()
        self.vecs = {}  # dim of vector space at node
        self.mats = {}  # linear maps (symbolic matrices)
        self.singularities = [] # used in symbolic calculation to store singularities

    # --- Grid and edges ---
    def generate_nodes(self):
        nodes = []
        for coordinates in self.generate_coordinates():
            nodes.append(tuple(coordinates))
        return nodes

    def generate_coordinates(self, current_dim=0, current_coords=None):
        if current_coords is None:
            current_coords = []
        if current_dim == len(self.dimensions):
            return [tuple(current_coords)]
        coordinates = []
        for i in range(self.dimensions[current_dim]):
            new_coords = current_coords + [i]
            coordinates.extend(self.generate_coordinates(current_dim + 1, new_coords))
        return coordinates

    def generate_edges(self):
        edges = []
        for node1 in self.nodes:
            for node2 in self.nodes:
                if all(node1[i] <= node2[i] for i in range(len(self.dimensions))):
                    edges.append((node1, node2))
        return edges

    # --- Vector spaces and matrices ---
    def create_vecs(self, node, m):
        if node in self.nodes:
            self.vecs[node] = m
        else:
            print("Error: node not in grid.")

    def create_matrix(self, node1, node2, matrix):
        if (node1, node2) in self.edges:
            u = self.vecs.get(node1, 0)
            v = self.vecs.get(node2, 0)
            if matrix is None:
                # store a zero matrix of the right size
                matrix = sp.zeros(v, u)
            else:
                if matrix.shape != (v, u):
                    print(f"Error: matrix dimensions mismatch for edge {node1}->{node2}. Expected ({v},{u}), got {matrix.shape}.")
                    return
            self.mats[(node1, node2)] = matrix
        else:
            print(f"Error: nodes {node1} and {node2} are not connected.")

    # --- Lattice operations ---
    def join(self, node1, node2):
        return tuple(max(a, b) for a, b in zip(node1, node2))

    def meet(self, node1, node2):
        return tuple(min(a, b) for a, b in zip(node1, node2))

    def is_smaller(self, node1, node2):
        return all(node1[i] <= node2[i] for i in range(len(self.dimensions)))

    # --- Convex hull of an interval ---
    def convex_square(self, node1, node2):
        if not self.is_smaller(node1, node2):
            return []
        points = []
        for coordinates in self.nodes:
            if all(node1[i] <= coordinates[i] <= node2[i] for i in range(len(self.dimensions))):
                points.append(tuple(coordinates))
        return points

    def int_hull(self, interval):
        points = []
        for src in interval.src:
            for snk in interval.snk:
                points += self.convex_square(src, snk)
        points = list(set(points))
        for pt in points[:]:
            if self.to_be_removed(pt, interval):
                points.remove(pt)
        return points

    def to_be_removed(self, point, interval):
        for src in interval.src:
            if self.is_smaller(point, src) and point != src:
                return True
        for snk in interval.snk:
            if self.is_smaller(snk, point) and point != snk:
                return True
        return False

    # --- Sources and sinks ---
    def get_src_snk(self, points):
        src, snk = [], []
        for pt in points:
            if self.add_source(pt, points):
                src.append(pt)
            if self.add_sink(pt, points):
                snk.append(pt)
        return src, snk

    def add_source(self, pt, points):
        for other in points:
            if self.is_smaller(other, pt) and other != pt:
                return False
        return True

    def add_sink(self, pt, points):
        for other in points:
            if self.is_smaller(pt, other) and other != pt:
                return False
        return True

##########################################
# --- New feature, Liu ---
    # --- Compute the dimension of an interval representation ---
    def int_dim(self, interval):
        return len(self.int_hull(interval))

    # # --- Compute the dimension vector of an interval representation ---
    # def int_dimvec(self, points, gridsize):
    #     rows, cols = gridsize
    #     dimvec = sp.zeros(rows, cols)
    #     for r, c in points:
    #         if 0 <= r < rows and 0 <= c < cols:
    #             dimvec[r, c] = 1

    #     for i in range(rows - 1, -1, -1):
    #         array_str = "".join(str(dimvec[i, j]) for j in range(cols))
    #         print(array_str)

    # --- Compute the dimension of the input representation ---
    def dim(self):
        return sum(self.vecs[node] for node in self.nodes)
 
    # --- Find proper sources and sinks ---
    def get_prop_src_snk(self, points):
        '''
        For each interval I, compute the source (resp. sink) of the proper up-set (down-set) of I 
        '''
        src_temp, snk_temp =  self.get_src_snk(points)
        upset = Interval(src_temp, [self.nodes[-1]])
        downset = Interval([self.nodes[0]], snk_temp)
        prop_up = [x for x in self.int_hull(upset) if x not in points]
        prop_down = [x for x in self.int_hull(downset) if x not in points]
        prp_src = self.get_src_snk(prop_up)[0]
        prp_snk = self.get_src_snk(prop_down)[1]
        return prp_src, prp_snk

    # --- Convert intervals of commutative ladders to dots or lines for the visualization purpose ---
    def dot_line_trans(self, interval):
        '''
        For each interval I of commutative ladders, spliting interval types and outputing the information of endpoints
        '''
        points = self.int_hull(interval)

        vals_0 = [v for c, v in points if c == 0]
        vals_1 = [v for c, v in points if c == 1]
        if vals_0 and not vals_1:
            return [(max(vals_0) + 1, min(vals_0))]
        elif vals_1 and not vals_0:
            return [(min(vals_1), max(vals_1) + 1)]
        else:
            tuple_0 = (max(vals_0) + 1, min(vals_0))
            tuple_1 = (min(vals_1), max(vals_1) + 1)
            return [tuple_0, tuple_1]
##########################################
# --- New feature (int_apprx), Liu ---
    # # --- List rays of fixed nodes ---
    # def get_horizontal_right_ray_nodes(self, node, bound):
    #     """
    #     Return a list of nodes having the same 1st coordinate as 'node',
    #     but with 2nd coordinate >= node[1], up to the dimension limit 'bound'.
    #     """
    #     x, start_y = node
    #     # Generate nodes: (x, y) for y in [start_y, bound - 1]
    #     return [(x, y) for y in range(start_y, bound)]

    # def get_horizontal_left_ray_nodes(self, node):
    #     """
    #     Return a list of nodes having the same 1st coordinate as 'node',
    #     but with 2nd coordinate <= node[1], down to 0.
    #     """
    #     x, start_y = node
    #     return [(x, y) for y in range(0, start_y + 1)]

    # def get_vertical_up_ray_nodes(self, node, bound):
    #     """
    #     Return a list of nodes having the same 2nd coordinate as 'node',
    #     but with 1st coordinate >= node[0], up to the dimension limit 'bound'.
    #     """
    #     start_x, y = node
    #     # Generate nodes: (x, y) for x in [start_x, bound - 1]
    #     return [(x, y) for x in range(start_x, bound)]

    # def get_vertical_down_ray_nodes(self, node):
    #     """
    #     Return a list of nodes having the same 2nd coordinate as 'node',
    #     but with 1st coordinate <= node[0], up to the dimension limit.
    #     """
    #     start_x, y = node
    #     return [(x, y) for x in range(0, start_x + 1)]

    # def out_of_grid(self, node):
    #     dimensions_grid = self.dimensions
    #     if node[0] not in range(0, dimensions_grid[0]) or node[1] not in range(0, dimensions_grid[1]):
    #         return True
    #     else:
    #         return False
        

    # # --- Find proper sources ---
    # def get_prop_src(self, points):
    #     '''
    #     For each interval I, compute the source of the proper up-set of I 
    #     '''
    #     src_temp =  self.get_src_snk(points)[0]
    #     upset = Interval(src_temp, [self.nodes[-1]])
    #     prop_up = [x for x in self.int_hull(upset) if x not in points]
    #     prp_src = self.get_src_snk(prop_up)[0]
    #     return prp_src

    # # --- Find proper sinks ---
    # def get_prop_snk(self, points):
    #     '''
    #     For each interval I, compute the sink of the proper down-set of I 
    #     '''
    #     snk_temp =  self.get_src_snk(points)[1]
    #     downset = Interval([self.nodes[0]], snk_temp)
    #     prop_down = [x for x in self.int_hull(downset) if x not in points]
    #     prp_snk = self.get_src_snk(prop_down)[1]
    #     return prp_snk

    # # --- Determine whether an element is in given interval ---
    # def is_in_interval(self, node, interval):
    #     '''
    #     Determine whether node is in interval by logic only, without generating the hull.
    #     '''
    #     # 1. Check whether the node lies in the convex-hull range, matching convex_square logic.
    #     # It must be greater than at least one source and smaller than at least one sink.
    #     larger_than_src = any(self.is_smaller(src, node) for src in interval.src)
    #     smaller_than_snk = any(self.is_smaller(node, snk) for snk in interval.snk)
    #     if not (larger_than_src and smaller_than_snk):
    #         return False

    #     # 2. Check whether the node should be removed, matching to_be_removed logic.
    #     # It must not be strictly smaller than any source; if node < src and node != src, it is outside the interval.
    #     for src in interval.src:
    #         if self.is_smaller(node, src) and node != src:
    #             return False

    #     # It must not be strictly greater than any sink.
    #     for snk in interval.snk:
    #         if self.is_smaller(snk, node) and node != snk:
    #             return False

    #     return True

    # # --- Radical approximations ---
    # def right_rad_apprx_inj(self, interval):
    #     '''
    #     Find right radical approximations of a given interval where morphisms are injective
    #     '''
    #     points = self.int_hull(interval)
    #     prp_snk = self.get_prop_snk(points)
    #     x_bound = self.dimensions[0]
    #     y_bound = self.dimensions[1]
    #     right_rad_apprx_inj = []

    #     for node in prp_snk:
    #         cant_move_up = self.is_in_interval(moveup(node), interval)
    #         cant_move_right = self.is_in_interval(moveright(node), interval)
            
    #         if cant_move_up and cant_move_right:
    #             new_interval_hull = points + [node]
    #             right_rad_apprx_inj.append(new_interval_hull)

    #         elif cant_move_up and not cant_move_right:
    #             new_interval_hull = points + self.get_horizontal_right_ray_nodes(node, y_bound)
    #             right_rad_apprx_inj.append(new_interval_hull)

    #         elif cant_move_right and not cant_move_up:
    #             new_interval_hull = points + self.get_vertical_up_ray_nodes(node, x_bound)
    #             right_rad_apprx_inj.append(new_interval_hull)
                
    #     return right_rad_apprx_inj

    # def right_rad_apprx_surj(self, interval):
    #     '''
    #     Find right radical approximations of a given interval where morphisms are surjective
    #     '''
    #     points = self.int_hull(interval)
    #     points_set = set(points)
    #     snk = interval.snk
    #     src = interval.src
    #     x_bound = self.dimensions[0]
    #     y_bound = self.dimensions[1]
    #     right_rad_apprx_surj = []

    #     for node in snk:
    #         x_coord = node[0]
    #         y_coord = node[1]
    #         exists_in_int_leftdown = self.is_in_interval(moveleftdown(node), interval) or self.out_of_grid(moveleftdown(node))
    #         exists_in_int_left = self.is_in_interval(moveleft(node), interval) or self.out_of_grid(moveleft(node))
    #         exists_in_int_down = self.is_in_interval(movedown(node), interval) or self.out_of_grid(movedown(node))

    #         if not exists_in_int_leftdown and not exists_in_int_left and not exists_in_int_down:
    #             pass
    #         elif not exists_in_int_leftdown and not exists_in_int_left and exists_in_int_down:
    #             pass
    #         elif not exists_in_int_leftdown and not exists_in_int_down and exists_in_int_left:
    #             pass
    #         elif not exists_in_int_leftdown and exists_in_int_down and exists_in_int_left:
    #             should_remove_vertical = (self.is_in_interval((0, y_coord), interval) and not self.is_in_interval((0, y_coord + 1), interval))
    #             should_remove_horizontal = (self.is_in_interval((x_coord, 0), interval) and not self.is_in_interval((x_coord + 1, 0), interval))

    #             if should_remove_vertical:
    #                 removal_nodes_v = self.get_vertical_down_ray_nodes(node)
    #                 new_interval_hull_v = list(points_set - set(removal_nodes_v))
    #                 right_rad_apprx_surj.append(new_interval_hull_v)

    #             if should_remove_horizontal:
    #                 removal_nodes_h = self.get_horizontal_left_ray_nodes(node)
    #                 new_interval_hull_h = list(points_set - set(removal_nodes_h))
    #                 right_rad_apprx_surj.append(new_interval_hull_h) 
    #             # if self.is_in_interval((0, y_coord), interval) and not self.is_in_interval((0, y_coord + 1), interval):
    #             #     removal_nodes = self.get_vertical_down_ray_nodes(node)
    #             #     new_interval_hull = list(points_set - set(removal_nodes))
    #             #     right_rad_apprx_surj.append(new_interval_hull)
    #             # if self.is_in_interval((x_coord, 0), interval) and not self.is_in_interval((x_coord + 1, 0), interval):
    #             #     removal_nodes = self.get_horizontal_left_ray_nodes(node)
    #             #     new_interval_hull = list(points_set - set(removal_nodes))
    #             #     right_rad_apprx_surj.append(new_interval_hull)
    #         else:
    #             new_interval_hull = list(points_set - set([node]))
    #             if len(new_interval_hull) != 0:
    #                 right_rad_apprx_surj.append(new_interval_hull)
    #             else:
    #                 pass

    #     return right_rad_apprx_surj
                
##########################################   

    # --- Evaluation of a path ---
    def evaluation(self, node1, node2):
        # handle empty vector spaces
        dim_out = self.vecs.get(node2, 0)
        dim_in = self.vecs.get(node1, 0)
        if dim_out == 0 or dim_in == 0:
            return sp.zeros(dim_out, dim_in)

        # start with identity matrix of appropriate size
        mat = sp.eye(dim_out)
        current = list(node2)
        previous = list(node2)

        for i in range(len(self.dimensions)):
            k = current[i] - node1[i]
            while k > 0:
                previous[i] -= 1
                edge = (tuple(previous), tuple(current))
                if self.mats.get(edge) is not None:
                    mat = mat * self.mats[edge]  # symbolic multiplication
                else:
                    return sp.zeros(dim_out, dim_in)
                current[i] -= 1
                k -= 1

        return mat


    # --- Construct block matrices ---
    def construct_matrix_MN(self, column_labels, block_signature, dual = False):
        '''
        Construct a matrix according to the block_signature: a list with elements [row, [col1, col2]].
        For each we add a block row with M_{col1, row} and -M_{col2, row}. The parameter column_labels
        determines the order of the column blocks.
        If dual is True, we add M_{row, col1}^T and -M_{row, col2}^T, and return transpose.
        '''
        col_num = sum(self.vecs[node] for node in column_labels)
        row_num = sum(self.vecs[node] for node, _ in block_signature)

        idx_col = {}  # column indices where each block should start and end
        start = 0
        for node in column_labels:
            idx_col[node] = (start, start + self.vecs[node])
            start = idx_col[node][1]

        matrix = sp.zeros(row_num, col_num)
        r_current, r_next = 0, 0  # current row

        for row, columns in block_signature:
            r_next = r_current + self.vecs[row]
            for i, col in enumerate(columns):
                block = self.evaluation(col, row) if not dual else self.evaluation(row, col).T
                matrix[r_current : r_next, idx_col[col][0] : idx_col[col][1]] = (-1)**i * block
            r_current = r_next

        return matrix if not dual else matrix.T


    def construct_matrix_M_tot(self, interval):
        '''Given an interval with n sources, return the matrix_M.'''
        column_labels = interval.src
        block_signature = []
        for i, src1 in enumerate(interval.src):
            for src2 in interval.src[i + 1:]:
                block_signature.append((self.join(src1, src2), (src1, src2)))
        return self.construct_matrix_MN(column_labels, block_signature)


    def construct_matrix_N_tot(self, interval):
        '''Given an interval with n sinks, return the matrix_N.'''
        column_labels = interval.snk
        block_signature = []
        for i, snk1 in enumerate(interval.snk):
            for snk2 in interval.snk[i + 1:]:
                block_signature.append((self.meet(snk1, snk2), (snk1, snk2)))
        return self.construct_matrix_MN(column_labels, block_signature, dual=True)


    def construct_matrix_M_ss(self, interval):
        block_signature = [(bound, (src1, src2))
                           for i, src1 in enumerate(interval.src)
                           for src2 in interval.src[i+1:]
                           for bound in self.find_upper_bounds(interval, src1, src2)]
        return self.construct_matrix_MN(interval.src, block_signature)

    def construct_matrix_N_ss(self, interval):
        block_signature = [(bound, (snk1, snk2))
                           for i, snk1 in enumerate(interval.snk)
                           for snk2 in interval.snk[i+1:]
                           for bound in self.find_lower_bounds(interval, snk1, snk2)]
        return self.construct_matrix_MN(interval.snk, block_signature, dual=True)

    def find_source_sink_indices_with_path(self, interval):
        for i in range(len(interval.src)):
            for j in range(len(interval.snk)):
                if self.is_smaller(interval.src[i], interval.snk[j]):
                    return i, j
        raise ValueError("No source -> sink path found.")
    
    def find_upper_bounds(self, interval, node1, node2):
        """
        Return a list of all sinks of the given interval larger than both given nodes.
        """
        return [snk for snk in interval.snk if self.is_smaller(node1, snk) and self.is_smaller(node2, snk)]

    def find_lower_bounds(self, interval, node1, node2):
        """
        Return a list of all sources of the given interval smaller than both given nodes.
        """
        return [src for src in interval.src if self.is_smaller(src, node1) and self.is_smaller(src, node2)]

    def list_int(self, conv=True):
        '''
        Return the list of all intervals possible.

        If conv == True, return intervals in the form of convex hulls.
        If conv == False, return intervals in the form Interval(src, snk).
        '''
        nodes = self.nodes
        candidates = powerset(nodes)[1:]  # remove the empty interval
        list_int = []
        for c in candidates:
            if self.is_convex(c) and self.is_connected(c):
                if conv:
                    list_int.append(c)
                else:
                    tmp = self.get_src_snk(c)
                    list_int.append(Interval(tmp[0], tmp[1]))
        return list_int

    def int_rank(self, interval, compression='tot'):
        i, j = self.find_source_sink_indices_with_path(interval)
        new_src, new_snk = interval.src.copy(), interval.snk.copy()
        new_src[0], new_src[i] = new_src[i], new_src[0]
        new_snk[0], new_snk[j] = new_snk[j], new_snk[0]
        interval = Interval(new_src, new_snk)

        if compression == 'tot':
            M = self.construct_matrix_M_tot(interval)
            N = self.construct_matrix_N_tot(interval)
            mat = self.evaluation(interval.src[0], interval.snk[0])
        elif compression == 'ss':
            M = self.construct_matrix_M_ss(interval)
            N = self.construct_matrix_N_ss(interval)
            mat = self.evaluation(interval.src[0], interval.snk[0])
        else:
            raise ValueError("Invalid compression")

        M_rows, M_cols = M.rows if M.rows > 0 else 1, M.cols if M.cols > 0 else 1
        N_rows, N_cols = N.rows if N.rows > 0 else 1, N.cols if N.cols > 0 else 1

        # Safe C matrix (match mat shape)
        C = sp.zeros(max(mat.rows, N_rows), max(mat.cols, M_cols))
        C[:mat.rows, :mat.cols] = mat

        # Safe zero blocks
        M_safe = M if M.rows > 0 and M.cols > 0 else sp.zeros(M_rows, M_cols)
        N_safe = N if N.rows > 0 and N.cols > 0 else sp.zeros(N_rows, N_cols)
        zeros_top_right = sp.zeros(M_rows, N_cols)

        # Construct block matrix safely
        block = sp.BlockMatrix([[M_safe, zeros_top_right], [C, N_safe]]).as_explicit()

        def symbolic_rank(matrix):
            """
            Compute the symbolic rank of a (possibly rectangular) SymPy matrix.
            Returns a sympy expression (Piecewise) depending on symbolic parameters.
            """
            rows, cols = matrix.shape
            if rows == 0 or cols == 0:
                return 0

            # The rank is at most min(rows, cols)
            max_rank = min(rows, cols)

            # Check all square minors from largest to smallest
            for k in range(max_rank, 0, -1):
                # Generate all sets of k rows and k columns
                from itertools import combinations
                row_combinations = list(combinations(range(rows), k))
                col_combinations = list(combinations(range(cols), k))

                for r_idx in row_combinations:
                    for c_idx in col_combinations:
                        minor = matrix.extract(list(r_idx), list(c_idx))
                        det = minor.det().simplify()
                        if not det.is_zero:
                            return sp.Piecewise((k, sp.Ne(det, 0)), (0, True))
            return 0

        return symbolic_rank(block) - symbolic_rank(M_safe) - symbolic_rank(N_safe)

##########################################
# --- New feature, Liu ---
    # --- Choice map for proper sources and sinks --- 
    def find_source_indices_per_proper_src(self, interval):
        """
        For each i in proper sources, find the first j in interval.src such that is_smaller(src[j], prp_src[i]) is True.
        Returns a list choice_src, or None if no such j exists for that i.
        """
        choice_src = []
        hull = self.int_hull(interval)
        prop_src_list = self.get_prop_src_snk(hull)[0]

        for i, prp_src_node in enumerate(prop_src_list):
            for j, src_node in enumerate(interval.src):
                if self.is_smaller(src_node, prp_src_node):
                    choice_src.append(j)
                    break
            else:
                # inner loop finished with no break
                raise ValueError(f"No source found for proper source index i={i}.")

        return choice_src

    def find_sink_indices_per_proper_snk(self, interval):
        """
        For each i in proper sinks, find the first j in interval.snk such that is_smaller(prp_snk[i], snk[j]) is True.
        Returns a list choice_snk, or None if no such j exists for that i.
        """
        choice_snk = []
        hull = self.int_hull(interval)
        prop_snk_list = self.get_prop_src_snk(hull)[1]

        for i, prp_snk_node in enumerate(prop_snk_list):
            for j, snk_node in enumerate(interval.snk):
                if self.is_smaller(prp_snk_node, snk_node):
                    choice_snk.append(j)
                    break
            else:
                # inner loop finished with no break
                raise ValueError(f"No sink found for proper sink index i={i}.")

        return choice_snk

    # --- Construct block matrices by using choice maps --- 
    def construct_matrix_M_choice(self, interval):
        """
        Construct the block matrix M_choice for computing interval multiplicity
        """
        column_labels = interval.src
        col_num = sum(self.vecs[node] for node in column_labels)
        
        hull = self.int_hull(interval)
        prop_src_labels = self.get_prop_src_snk(hull)[0]
        choice_src = self.find_source_indices_per_proper_src(interval)

        # Interval does not have proper sources, return a zero matrix with size 1 × col_num
        if not prop_src_labels:
            return sp.zeros(1, col_num)

        # Interval has proper sources
        row_labels = prop_src_labels
        row_num = sum(self.vecs[node] for node in row_labels)
        M_choice = sp.zeros(row_num, col_num)

        # Compute (start, end)-position of each column block in the bigger matrix
        idx_col = {}
        c_start = 0
        for c in column_labels:
            width = self.vecs[c]
            idx_col[c] = (c_start, c_start + width)
            c_start += width

        # Replace corresponding zero blocks following row indexes and the choice map
        r_current = 0
        for i, prp_src_node in enumerate(row_labels):
            height = self.vecs[prp_src_node]
            r_next = r_current + height

            j = choice_src[i]
            src_node = column_labels[j]
            c_start, c_end = idx_col[src_node]

        # replace corresponding zero blocks
            block = self.evaluation(src_node, prp_src_node)
            M_choice[r_current:r_next, c_start:c_end] = block
            r_current = r_next

        return M_choice

    def construct_matrix_N_choice(self, interval):
        """
        Construct the block matrix N_choice for computing interval multiplicity
        """
        row_labels = interval.snk
        row_num = sum(self.vecs[node] for node in row_labels)
        
        hull = self.int_hull(interval)
        prop_snk_labels = self.get_prop_src_snk(hull)[1]
        choice_snk = self.find_sink_indices_per_proper_snk(interval)

        # Interval does not have proper sinks, return a zero matrix with size row_num × 1
        if not prop_snk_labels:
            return sp.zeros(row_num, 1)

        # Interval has proper sinks
        column_labels = prop_snk_labels
        col_num = sum(self.vecs[node] for node in column_labels)
        N_choice = sp.zeros(row_num, col_num)

        # Compute (start, end)-position of each row block in the bigger matrix
        idx_row = {}
        r_start = 0
        for r in row_labels:
            width = self.vecs[r]
            idx_row[r] = (r_start, r_start + width)
            r_start += width

        # Replace corresponding zero blocks following column indexes and the choice map
        c_current = 0
        for i, prp_snk_node in enumerate(column_labels):
            height = self.vecs[prp_snk_node]
            c_next = c_current + height

            j = choice_snk[i]
            snk_node = row_labels[j]
            r_start, r_end = idx_row[snk_node]

        # replace corresponding zero blocks
            block = self.evaluation(prp_snk_node, snk_node)
            N_choice[r_start:r_end, c_current:c_next] = block
            c_current = c_next

        return N_choice

    # --- Interval multiplicity --- 
    def int_mult(self, interval):
        i, j = self.find_source_sink_indices_with_path(interval)
        new_src, new_snk = interval.src.copy(), interval.snk.copy()
        new_src[0], new_src[i] = new_src[i], new_src[0]
        new_snk[0], new_snk[j] = new_snk[j], new_snk[0]
        interval = Interval(new_src, new_snk)
        M_pair = self.construct_matrix_M_tot(interval)
        M_choice = self.construct_matrix_M_choice(interval)
        N_pair = self.construct_matrix_N_tot(interval)
        N_choice = self.construct_matrix_N_choice(interval)
        mat = self.evaluation(interval.src[0], interval.snk[0])

        # Stack with choice matrices
        M_safe = sp.BlockMatrix([[M_pair], [M_choice]]).as_explicit()
        N_safe = sp.BlockMatrix([[N_choice, N_pair]]).as_explicit()
        
        # Create the top-right zero matrix
        zeros_top_right = sp.zeros(M_safe.rows, N_safe.cols)

        # Create C

        total_rows = N_safe.rows
        total_cols = M_safe.cols

        C = sp.zeros(total_rows, total_cols)

        if mat.rows > 0 and mat.cols > 0:
            # Ensure mat does not exceed the shape of C. This should not happen because mat is local and C is global.
            if mat.rows <= total_rows and mat.cols <= total_cols:
                C[:mat.rows, :mat.cols] = mat
            else:
                 # Defensive fallback: if mat is larger than the total dimensions, there is a deeper logic error.
                pass

        # M_pair_rows, M_pair_cols = M_pair.rows if M_pair.rows > 0 else 1, M_pair.cols if M_pair.cols > 0 else 1
        # N_pair_rows, N_pair_cols = N_pair.rows if N_pair.rows > 0 else 1, N_pair.cols if N_pair.cols > 0 else 1

        # # Safe C matrix (match mat shape)
        # C = sp.zeros(max(mat.rows, N_pair_rows), max(mat.cols, M_pair_cols))
        # C[:mat.rows, :mat.cols] = mat

        # # Safe zero blocks
        # M_pair_safe = M_pair if M_pair.rows > 0 and M_pair.cols > 0 else sp.zeros(M_pair_rows, M_pair_cols)
        # N_pair_safe = N_pair if N_pair.rows > 0 and N_pair.cols > 0 else sp.zeros(N_pair_rows, N_pair_cols)

        

        # Construct block matrix safely
        block = sp.BlockMatrix([[M_safe, zeros_top_right], [C, N_safe]]).as_explicit()

        def symbolic_rank(matrix):
            """
            Compute the symbolic rank of a (possibly rectangular) SymPy matrix.
            Returns a sympy expression (Piecewise) depending on symbolic parameters.
            """
            rows, cols = matrix.shape
            if rows == 0 or cols == 0:
                return 0

            # The rank is at most min(rows, cols)
            max_rank = min(rows, cols)

            # Check all square minors from largest to smallest
            for k in range(max_rank, 0, -1):
                # Generate all sets of k rows and k columns
                from itertools import combinations
                row_combinations = list(combinations(range(rows), k))
                col_combinations = list(combinations(range(cols), k))

                for r_idx in row_combinations:
                    for c_idx in col_combinations:
                        minor = matrix.extract(list(r_idx), list(c_idx))
                        det = minor.det().simplify()
                        if not det.is_zero:
                            return sp.Piecewise((k, sp.Ne(det, 0)), (0, True))
            return 0

        return symbolic_rank(block) - symbolic_rank(M_safe) - symbolic_rank(N_safe)
##########################################   

    # --- Interval replacement ---
    def int_replacement(self, interval, compression='tot'):
        repl = self.int_rank(interval, compression)
        cov_ps = powerset(self.cover(interval))
        for c in cov_ps[1:]:
            if len(c) == 1:
                tmp = self.get_src_snk(c[0])
                i = Interval(tmp[0], tmp[1])
                repl -= self.int_rank(i, compression)
            if len(c) > 1:
                eps = len(c)
                c_flat = [list(set(flatten(c)))]
                tmp = self.get_src_snk(c_flat[0])
                i2 = Interval(tmp[0], tmp[1])
                repl += (-1)**eps * self.int_rank(i2, compression)
        return(repl)

    def cover(self, interval, conv=True):
        """Compute the cover of an interval without duplicates."""
        cover_list = []
        hull = self.int_hull(interval)
        seen = set()  # Track seen intervals to avoid duplicates
        
        for point in hull:
            if self.is_corner(point, interval):
                # Try extending in each dimension
                for i in range(len(self.dimensions)):
                    # Try increasing coordinate
                    if point[i] < self.dimensions[i] - 1:
                        new_point = list(point)
                        new_point[i] += 1
                        new_point = tuple(new_point)
                        
                        if new_point not in hull:
                            # Create new hull by adding the new point
                            new_hull = hull + [new_point]
                            if self.is_convex(new_hull):
                                if conv:
                                    # For point lists: use sorted tuple as signature
                                    signature = tuple(sorted(new_hull))
                                    if signature not in seen:
                                        cover_list.append(new_hull)
                                        seen.add(signature)
                                else:
                                    # For Interval objects: get sources/sinks
                                    tmp = self.get_src_snk(new_hull)
                                    new_interval = Interval(tmp[0], tmp[1])
                                    sig = (tuple(sorted(new_interval.src)), tuple(sorted(new_interval.snk)))
                                    if sig not in seen:
                                        cover_list.append(new_interval)
                                        seen.add(sig)
                    
                    # Try decreasing coordinate  
                    if point[i] > 0:
                        new_point = list(point)
                        new_point[i] -= 1
                        new_point = tuple(new_point)
                        
                        if new_point not in hull:
                            # Create new hull by adding the new point
                            new_hull = hull + [new_point]
                            if self.is_convex(new_hull):
                                if conv:
                                    # For point lists: use sorted tuple as signature
                                    signature = tuple(sorted(new_hull))
                                    if signature not in seen:
                                        cover_list.append(new_hull)
                                        seen.add(signature)
                                else:
                                    # For Interval objects: get sources/sinks
                                    tmp = self.get_src_snk(new_hull)
                                    new_interval = Interval(tmp[0], tmp[1])
                                    sig = (tuple(sorted(new_interval.src)), tuple(sorted(new_interval.snk)))
                                    if sig not in seen:
                                        cover_list.append(new_interval)
                                        seen.add(sig)
        
        return cover_list

    def is_convex(self, points):
        tmp = self.get_src_snk(points)
        interval = Interval(tmp[0], tmp[1])
        return set(points) == set(self.int_hull(interval))

    def is_corner(self, point, interval):
        for i in range(len(self.dimensions)):
            aligned = False
            for src in interval.src:
                if src[i] == point[i]:
                    aligned = True
            for snk in interval.snk:
                if snk[i] == point[i]:
                    aligned = True
            if not aligned:
                return False
        return True

    def is_connected(self, points):
        visited = set()
        def dfs(node):
            visited.add(node)
            for neighbor in points:
                if neighbor not in visited and (self.is_smaller(node, neighbor) or self.is_smaller(neighbor, node)):
                    dfs(neighbor)
        dfs(points[0])
        return all(node in visited for node in points)
    
    # --- Check validity of symbolic calculation ---
    def find_complete_singularities(self, parameter):
        """
        Find all singular parameter values where interval replacement computations may change.
        This identifies where symbolic computation cannot be simply substituted.
        """
        print("Identifying singular parameter values for interval replacement")
        
        all_intervals = self.list_int(conv=False)
        print(f"Total intervals in quiver: {len(all_intervals)}")
        
        all_replacement_intervals = set()
        
        for interval in all_intervals:
            all_replacement_intervals.add((tuple(sorted(interval.src)), tuple(sorted(interval.snk))))
            
            cover_intervals = self.cover(interval, conv=False)
            for cov in cover_intervals:
                all_replacement_intervals.add((tuple(sorted(cov.src)), tuple(sorted(cov.snk))))
            
            cov_ps = powerset(cover_intervals)
            for c in cov_ps:
                if len(c) > 1:
                    all_points = []
                    for interval_obj in c:
                        hull_points = self.int_hull(interval_obj)
                        all_points.extend(hull_points)
                    unique_points = list(set(all_points))
                    tmp = self.get_src_snk(unique_points)
                    combined_int = Interval(tmp[0], tmp[1])
                    all_replacement_intervals.add((tuple(sorted(combined_int.src)), tuple(sorted(combined_int.snk))))
        
        print(f"Total unique intervals in replacement formulas: {len(all_replacement_intervals)}")
        
        complete_singularities = set()
        
        def construct_block_matrix(rep, src, snk):
            interval = Interval(list(src), list(snk))
            i, j = rep.find_source_sink_indices_with_path(interval)
            new_src, new_snk = interval.src.copy(), interval.snk.copy()
            new_src[0], new_src[i] = new_src[i], new_src[0]
            new_snk[0], new_snk[j] = new_snk[j], new_snk[0]
            interval = Interval(new_src, new_snk)

            M = rep.construct_matrix_M_ss(interval)
            N = rep.construct_matrix_N_ss(interval)
            mat = rep.evaluation(interval.src[0], interval.snk[0])

            M_rows, M_cols = M.rows if M.rows > 0 else 1, M.cols if M.cols > 0 else 1
            N_rows, N_cols = N.rows if N.rows > 0 else 1, N.cols if N.cols > 0 else 1

            C = sp.zeros(max(mat.rows, N_rows), max(mat.cols, M_cols))
            C[:mat.rows, :mat.cols] = mat

            M_safe = M if M.rows > 0 and M.cols > 0 else sp.zeros(M_rows, M_cols)
            N_safe = N if N.rows > 0 and N.cols > 0 else sp.zeros(N_rows, N_cols)
            zeros_top_right = sp.zeros(M_rows, N_cols)

            block = sp.BlockMatrix([[M_safe, zeros_top_right], [C, N_safe]]).as_explicit()
            return block
        
        def find_singularities(matrix, param):
            rows, cols = matrix.shape
            max_rank = min(rows, cols)
            singularities = set()
            
            for k in range(max_rank, 0, -1):
                for row_indices in combinations(range(rows), k):
                    for col_indices in combinations(range(cols), k):
                        minor = matrix.extract(list(row_indices), list(col_indices))
                        det = minor.det().simplify()
                        
                        if det != 0:
                            solutions = sp.solve(det, param)
                            for sol in solutions:
                                try:
                                    singularities.add(float(sol.evalf()))
                                except:
                                    singularities.add(sol)
                            return singularities
            return singularities
        
        for i, (src, snk) in enumerate(all_replacement_intervals):
            if i % 20 == 19:
                print(f"Progress: {i+1}/{len(all_replacement_intervals)}")
                
            try:
                block = construct_block_matrix(self, src, snk)
                singularities = find_singularities(block, parameter)
                complete_singularities.update(singularities)
            except:
                continue
        
        print(f"Singular parameter values: {complete_singularities}")

        self.singularities = complete_singularities
        
        return complete_singularities
    
    def has_symbols(self):
        """Return True if the representation contains any symbolic variables, False otherwise."""
        # Check matrices for symbols
        for matrix in self.mats.values():
            if matrix.has(sp.Symbol):
                return True
        
        # Check vector space dimensions for symbols
        for dim in self.vecs.values():
            if isinstance(dim, sp.Expr) and dim.has(sp.Symbol):
                return True
        
        return False

    # --- Utility ---
    def elements(self):
        print("Nodes:", self.nodes)
        print("Edges:", self.edges)
        print("Vector spaces:", self.vecs)
        print("Matrices:", self.mats)
