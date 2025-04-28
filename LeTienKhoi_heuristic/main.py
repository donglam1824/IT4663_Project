from ortools.sat.python import cp_model

def solve_2d_loading(N, K, item_sizes, truck_info):
    model = cp_model.CpModel()
    max_x = max([truck_info[k][0] for k in range(K)])  # Max width of all trucks
    max_y = max([truck_info[k][1] for k in range(K)])  # Max height of all trucks
    # === Variables ===
    t = [model.NewIntVar(0, K - 1, f"t_{i}") for i in range(N)]        # Truck index for item i
    o = [model.NewBoolVar(f"o_{i}") for i in range(N)]                 # Orientation
    x = [model.NewIntVar(0, max_x, f"x_{i}") for i in range(N)]         # x coordinate
    y = [model.NewIntVar(0, max_y, f"y_{i}") for i in range(N)]         # y coordinate
    used = [model.NewBoolVar(f"used_{k}") for k in range(K)]           # Whether truck k is used

    # is_truck[i][k] = True if item i is in truck k
    is_truck = [[model.NewBoolVar(f"is_truck_{i}_{k}") for k in range(K)] for i in range(N)]
    for i in range(N):
        model.AddExactlyOne(is_truck[i])
        for k in range(K):
            model.Add(t[i] == k).OnlyEnforceIf(is_truck[i][k])
            model.Add(t[i] != k).OnlyEnforceIf(is_truck[i][k].Not())

    # === Fit item inside truck bounds ===
    for i in range(N):
        w_i, l_i = item_sizes[i]
        for k in range(K):
            Wk, Lk, _ = truck_info[k]

            # If orientation is 0 (not rotated)
            model.Add(x[i] + w_i <= Wk).OnlyEnforceIf([o[i].Not(), is_truck[i][k]])
            model.Add(y[i] + l_i <= Lk).OnlyEnforceIf([o[i].Not(), is_truck[i][k]])

            # If orientation is 1 (rotated)
            model.Add(x[i] + l_i <= Wk).OnlyEnforceIf([o[i], is_truck[i][k]])
            model.Add(y[i] + w_i <= Lk).OnlyEnforceIf([o[i], is_truck[i][k]])

    # === No Overlap in same truck ===
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(K):
                w_i, l_i = item_sizes[i]
                w_j, l_j = item_sizes[j]

                # Create boolean: both in same truck k
                both_in_k = model.NewBoolVar(f"both_in_{i}_{j}_k{k}")
                model.AddBoolAnd([is_truck[i][k], is_truck[j][k]]).OnlyEnforceIf(both_in_k)
                model.AddBoolOr([is_truck[i][k].Not(), is_truck[j][k].Not()]).OnlyEnforceIf(both_in_k.Not())

                # 4 possible separation directions
                sep_x1 = model.NewBoolVar(f"sep_x1_{i}_{j}_k{k}")  # i on left of j
                sep_x2 = model.NewBoolVar(f"sep_x2_{i}_{j}_k{k}")  # j on left of i
                sep_y1 = model.NewBoolVar(f"sep_y1_{i}_{j}_k{k}")  # i below j
                sep_y2 = model.NewBoolVar(f"sep_y2_{i}_{j}_k{k}")  # j below i

                # Width/Height depends on orientation
                wi_0, li_0 = w_i, l_i
                wi_1, li_1 = l_i, w_i
                wj_0, lj_0 = w_j, l_j
                wj_1, lj_1 = l_j, w_j

                # Expressions for width/height depending on o[i]
                w_i_var = model.NewIntVar(1, 1000, f"w_i_{i}")
                l_i_var = model.NewIntVar(1, 1000, f"l_i_{i}")
                w_j_var = model.NewIntVar(1, 1000, f"w_j_{j}")
                l_j_var = model.NewIntVar(1, 1000, f"l_j_{j}")

                model.Add(w_i_var == wi_0).OnlyEnforceIf(o[i].Not())
                model.Add(w_i_var == wi_1).OnlyEnforceIf(o[i])
                model.Add(l_i_var == li_0).OnlyEnforceIf(o[i].Not())
                model.Add(l_i_var == li_1).OnlyEnforceIf(o[i])

                model.Add(w_j_var == wj_0).OnlyEnforceIf(o[j].Not())
                model.Add(w_j_var == wj_1).OnlyEnforceIf(o[j])
                model.Add(l_j_var == lj_0).OnlyEnforceIf(o[j].Not())
                model.Add(l_j_var == lj_1).OnlyEnforceIf(o[j])

                # Directions
                model.Add(x[i] + w_i_var <= x[j]).OnlyEnforceIf(sep_x1)
                model.Add(x[j] + w_j_var <= x[i]).OnlyEnforceIf(sep_x2)
                model.Add(y[i] + l_i_var <= y[j]).OnlyEnforceIf(sep_y1)
                model.Add(y[j] + l_j_var <= y[i]).OnlyEnforceIf(sep_y2)

                model.AddBoolOr([sep_x1, sep_x2, sep_y1, sep_y2]).OnlyEnforceIf(both_in_k)

    # === Mark used trucks ===
    for k in range(K):
        items_in_k = [is_truck[i][k] for i in range(N)]
        model.AddBoolOr(items_in_k).OnlyEnforceIf(used[k])
        model.AddBoolAnd([b.Not() for b in items_in_k]).OnlyEnforceIf(used[k].Not())

    # === Objective: Minimize cost ===
    total_cost = sum(used[k] * truck_info[k][2] for k in range(K))
    model.Minimize(total_cost)

    # === Solve ===
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [solver.Value(t[i]) for i in range(N)], \
               [solver.Value(x[i]) for i in range(N)], \
               [solver.Value(y[i]) for i in range(N)], \
               [solver.Value(o[i]) for i in range(N)]
    else:
        print("No solution found.")

class Solution:
    def __init__(self, trucks, x_coords, y_coords, orientations, cost_truck):
        self.trucks = trucks
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.orientations = orientations
        self.fitness = sum(cost_truck[k] for k in set(trucks))  # Tính toán chi phí dựa trên các xe tải đã sử dụng

    def __str__(self):
        return f"Trucks: {self.trucks}, X: {self.x_coords}, Y: {self.y_coords}, Orientations: {self.orientations}, Fitness: {self.fitness}"
    
def generate_neighbor(N, K, Solution, batch_size, item_sizes, truck_info):
    import random
    import copy

    copy_solution = copy.deepcopy(Solution)  # Tạo bản sao của lời giải hiện tại

    while True:  # Lặp cho đến khi chọn được sub_N < 15
        # Chọn ngẫu nhiên một số truck để tối ưu
        selected_trucks = random.sample(range(K), min(batch_size, K))
        selected_items = [i for i in range(N) if Solution.trucks[i] in selected_trucks]

        sub_N = len(selected_items)
        if sub_N < 15:  # Nếu thỏa mãn điều kiện, tiếp tục kiểm tra
            # Tính tổng diện tích các item
            sum_item_area = sum([item_sizes[i][0] * item_sizes[i][1] for i in selected_items])

            # Tìm xe có diện tích nhỏ nhất
            min_truck_area = min([truck_info[k][0] * truck_info[k][1] for k in selected_trucks])
            total_truck_area = sum([truck_info[k][0] * truck_info[k][1] for k in selected_trucks])

            # Kiểm tra nếu bỏ xe có diện tích nhỏ nhất mà tổng diện tích còn lại nhỏ hơn tổng diện tích hàng
            if total_truck_area - min_truck_area < sum_item_area:
                continue  # Không hợp lệ, chọn lại

            break  # Nếu hợp lệ, thoát khỏi vòng lặp

    sub_item_sizes = [item_sizes[i] for i in selected_items]
    sub_truck_info = [truck_info[k] for k in selected_trucks]
    sub_K = len(selected_trucks)

    # Thực hiện tối ưu nếu hợp lệ
    sub_t, sub_x, sub_y, sub_o = solve_2d_loading(sub_N, sub_K, sub_item_sizes, sub_truck_info)

    for i, item in enumerate(selected_items):
        copy_solution.trucks[item] = selected_trucks[sub_t[i]]  # cần ánh xạ lại nếu sub_t là chỉ số cục bộ
        copy_solution.x_coords[item] = sub_x[i]
        copy_solution.y_coords[item] = sub_y[i]
        copy_solution.orientations[item] = sub_o[i]

    # Cập nhật fitness
    copy_solution.fitness = sum(truck_info[k][2] for k in set(copy_solution.trucks))  # Cập nhật chi phí dựa trên các xe tải đã sử dụng

    return copy_solution

def generate_initial_solution_greedy(N, K, item_sizes, truck_info):
    # Sắp xếp truck theo chi phí tăng dần
    sorted_trucks = sorted([(k, *truck_info[k]) for k in range(K)], key=lambda x: x[3])  # (k, W, H, cost)

    trucks = [-1] * N
    x_coords = [0] * N
    y_coords = [0] * N
    orientations = [0] * N

    # Với mỗi truck, lưu danh sách các item đã đặt: (x, y, w, h)
    packed_items = {k: [] for k in range(K)}

    for i in range(N):
        w_i, h_i = item_sizes[i]
        placed = False

        for orientation in [0, 1]:
            wi = w_i if orientation == 0 else h_i
            hi = h_i if orientation == 0 else w_i

            for k, Wk, Hk, _ in sorted_trucks:
                # Thử từng vị trí lưới đơn giản (mỗi ô 10x10)
                for x in range(0, Wk - wi + 1, 10):
                    for y in range(0, Hk - hi + 1, 10):
                        overlap = False
                        for (px, py, pw, ph) in packed_items[k]:
                            if not (x + wi <= px or px + pw <= x or y + hi <= py or py + ph <= y):
                                overlap = True
                                break
                        if not overlap:
                            trucks[i] = k
                            x_coords[i] = x
                            y_coords[i] = y
                            orientations[i] = orientation
                            packed_items[k].append((x, y, wi, hi))
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break

        if not placed:
            print(f"⚠️ Warning: Item {i} không đặt được vào truck nào.")

    return Solution(trucks, x_coords, y_coords, orientations, [truck_info[k][2] for k in range(K)])

def Tabu_Search(N, K, item_sizes, truck_info, max_iterations=200, tabu_tenure=10):
    import copy

    # Khởi tạo lời giải ban đầu
    current_solution = generate_initial_solution_greedy(N, K, item_sizes, truck_info)
    best_solution = copy.deepcopy(current_solution)

    # Danh sách tabu
    tabu_list = []

    # Bắt đầu vòng lặp Tabu Search
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")

        # Sinh hàng xóm
        neighbors = []
        for _ in range(10):  # Tạo tối đa 10 hàng xóm
            neighbor = generate_neighbor(N, K, current_solution, 7, item_sizes, truck_info)
            if neighbor not in tabu_list:
                neighbors.append(neighbor)

        # Nếu không có hàng xóm hợp lệ, dừng thuật toán
        if not neighbors:
            print("No valid neighbors found. Stopping.")
            break

        # Chọn hàng xóm tốt nhất
        best_neighbor = min(neighbors, key=lambda sol: sol.fitness)

        # Cập nhật danh sách tabu
        tabu_list.append(copy.deepcopy(current_solution))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        # Cập nhật lời giải hiện tại
        current_solution = best_neighbor

        # Cập nhật lời giải tốt nhất nếu cần
        if current_solution.fitness < best_solution.fitness:
            best_solution = copy.deepcopy(current_solution)

        print(f"Current best fitness: {best_solution.fitness}")
        result_path = pathlib.Path(__file__).parent / "Result"/ "test10.txt"
        with open(result_path, "w") as result_file:
            for i in range(N):
                result_file.write(f"{i+1} {best_solution.trucks[i] + 1} {best_solution.x_coords[i]} {best_solution.y_coords[i]} {best_solution.orientations[i]}\n")
            result_file.write(f"Total cost: {best_solution.fitness}\n")

    return best_solution
def Tabu__Search_With_Start_Solution(N, K, item_sizes, truck_info, start_solution, max_iterations=200, tabu_tenure=10):
    import copy

    # Khởi tạo lời giải ban đầu
    current_solution = start_solution
    best_solution = copy.deepcopy(current_solution)

    # Danh sách tabu
    tabu_list = []

    # Bắt đầu vòng lặp Tabu Search
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")

        # Sinh hàng xóm
        neighbors = []
        for _ in range(10):  # Tạo tối đa 10 hàng xóm
            neighbor = generate_neighbor(N, K, current_solution, 5, item_sizes, truck_info)
            if neighbor not in tabu_list:
                neighbors.append(neighbor)

        # Nếu không có hàng xóm hợp lệ, dừng thuật toán
        if not neighbors:
            print("No valid neighbors found. Stopping.")
            break

        # Chọn hàng xóm tốt nhất
        best_neighbor = min(neighbors, key=lambda sol: sol.fitness)

        # Cập nhật danh sách tabu
        tabu_list.append(copy.deepcopy(current_solution))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        # Cập nhật lời giải hiện tại
        current_solution = best_neighbor

        # Cập nhật lời giải tốt nhất nếu cần
        if current_solution.fitness < best_solution.fitness:
            best_solution = copy.deepcopy(current_solution)

        print(f"Current best fitness: {best_solution.fitness}")
        result_path = pathlib.Path(__file__).parent / "Result"/ "test10.txt"
        with open(result_path, "w") as result_file:
            for i in range(N):
                result_file.write(f"{i+1} {best_solution.trucks[i] + 1} {best_solution.x_coords[i]} {best_solution.y_coords[i]} {best_solution.orientations[i]}\n")
            result_file.write(f"Total cost: {best_solution.fitness}\n")

    return best_solution

def get_start_solution_from_file(file_path,truck_info):
    trucks = []
    x_coords = []
    y_coords = []
    orientations = []
    cost_truck = [truck[2] for truck in truck_info]  # Chi phí của từng xe tải

    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()  # Loại bỏ khoảng trắng ở đầu và cuối dòng
            if not line:  # Bỏ qua dòng trống
                continue
            if line.startswith("Total cost:"):
                # Đọc tổng chi phí từ dòng cuối
                total_cost = int(line.split(":")[1].strip())
            else:
                # Đọc thông tin từng item
                parts = line.split()
                if len(parts) != 5:  # Kiểm tra nếu dòng không có đúng 5 phần tử
                    print(f"⚠️ Warning: Invalid line format: {line}")
                    continue
                try:
                    item_id = int(parts[0])  # ID của item (không cần sử dụng)
                    truck = int(parts[1]) - 1  # Truck index (giảm 1 để phù hợp với index 0-based)
                    x = int(parts[2])  # Tọa độ x
                    y = int(parts[3])  # Tọa độ y
                    orientation = int(parts[4])  # Orientation (0 hoặc 1)

                    trucks.append(truck)
                    x_coords.append(x)
                    y_coords.append(y)
                    orientations.append(orientation)
                except ValueError as e:
                    print(f"⚠️ Warning: Unable to parse line: {line}. Error: {e}")
                    continue
    print(trucks, x_coords, y_coords, orientations)
    # Tạo đối tượng Solution
    return Solution(trucks, x_coords, y_coords, orientations, cost_truck)

import sys
import pathlib

if __name__ == "__main__":
    parent_dir = pathlib.Path(__file__).parent.parent.resolve()
    re_path = pathlib.Path(__file__).parent / "Result"/ "test10.txt"
    testcase_path = parent_dir / "testcase" / "test10.txt"
    sys.stdin = open(testcase_path, "r")
    N, K = map(int, input().split())
    item_sizes = [tuple(map(int, input().split())) for _ in range(N)]
    truck_info = [tuple(map(int, input().split())) for _ in range(K)]

    star_solution = get_start_solution_from_file(re_path,truck_info)

    # optimized_solution = Tabu_Search(N, K, item_sizes, truck_info, max_iterations=200, tabu_tenure=10)
    optimized_solution = Tabu__Search_With_Start_Solution(N, K, item_sizes, truck_info, star_solution, max_iterations=20000, tabu_tenure=10)
    # for i in range(N):
    #     print(i+1, optimized_solution.trucks[i] + 1, optimized_solution.x_coords[i], optimized_solution.y_coords[i], optimized_solution.orientations[i])
    # print("Total cost:", optimized_solution.fitness)
