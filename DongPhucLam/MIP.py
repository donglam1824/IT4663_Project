import os
import time

from ortools.linear_solver import pywraplp

def input(testcase_path):
    data = {}

    with open(testcase_path, 'r') as file:
        lines = file.readlines()

    # n items, k trucks
    n, k = map(int, lines[0].split())

    data['size_item'] = []
    data['size_truck'] = []
    data['cost'] = []

    for i in range(n):
        w, h = map(int, lines[1 + i].split())   #width, height of items
        data['size_item'].append([w, h])

    for j in range(k):
        w, h, c = map(int, lines[1 + n + j].split())    #width, height, cost of trucks
        data['size_truck'].append([w, h])
        data['cost'].append(c)

    W_truck = [data['size_truck'][i][0] for i in range(k)]
    H_truck = [data['size_truck'][i][1] for i in range(k)]

    return n, k, data, W_truck, H_truck

def solve(testcase_path):

    n, k, data, W_truck, H_truck = input(testcase_path)

    size_item = data['size_item']
    size_truck = data['size_truck']
    cost = data['cost']
    
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solve:
        return None
    
    
    #Bien quyet dinh
    #Bien nhi phan u[k] = 1 neu xe duoc dung
    u = {}
    for k in range(k):
        u[k] = solver.IntVar(0, 1, f'u_{k}')

    #Bien nhi phan t[i][j] neu items i xep vao xe j
    t = {}
    for i in range (n):
        t[i] = {}
        for j in range (k):
            t[i][j] = solver.IntVar(0, 1, f't_{i}_{j}')

    #Bien nguyen: x[i], y[i] toa do goc duoi ben trai
    x = {}
    y = {}

    for i in range(n):
        x[i] = solver.IntVar(0, 1000, f'x_{i}')
        y[i] = solver.IntVar(0, 1000, f'y_{i}')

    #Bien nhi phan o[i] = 1 neu mon hang i duoc xoay 90 do
    o = {}

    for i in range(n):
        o[i] = solver.IntVar(0, 1000, f'o_{i}')   

    #Bien phu
    #Mon hang i co o ben trai/phai/duoi/tren mon hang j khong
    left = {}
    right = {}
    below = {}
    above = {}    

    big_M = 2000

    for i in range(n):
        for j in range(i+1, n):
            left[(i, j)] = solver.IntVar(0, 1, f'left_{i}_{j}')
            right[(i, j)] = solver.IntVar(0, 1, f'right_{i}_{j}')
            below[(i, j)] = solver.IntVar(0, 1, f'below_{i}_{j}')
            above[(i, j)] = solver.IntVar(0, 1, f'above_{i}_{j}')

    # Rang buoc
    # Moi mon chi xep vao mot xe

    for i in range(n):
        solver.Add(sum(t[i][j] for j in range(k)) == 1)

    # Xe tai duoc dung khi co it nhat mot mon
    # u[j] = 1 khi co it nhat t[i][j] = 1
    for j in range(k):
        for i in range(n):
            solver.Add(t[i][j] <= u[j])     

    # Hang phai nam trong kich thuoc xe
    for i in range(n):
        w_i = size_item[i][0]
        h_i = size_item[i][1]

        for j in range(k):
            W_j = size_truck[j][0]
            H_j = size_truck[j][1]

            # Neu item i duoc xep vao xe j
            # Rang buoc ve chieu ngang

            solver.Add(x[i] + w_i * (1 - o[i]) + h_i * o[i] <= W_j + (1 - t[i][j]) * big_M)

            # Rang buoc ve chieu rong

            solver.Add(y[i] + w_i * o[i] + h_i * (1 - o[i]) <= H_j + (1 - t[i][j] * big_M))

    # Hang khong duoc chong len nhau neu o cung xe
    for i in range(n):
        w_i = size_item[i][0]
        h_i = size_item[i][1]

        for j in range(i+1, n):
            w_j = size_item[j][0]
            h_j = size_item[j][1]

            for truck in range(k):
                same_truck = solver.IntVar(0, 1, f'same_truck_{i}_{j}_{truck}')
                solver.Add(same_truck >= t[i][truck] + t[j][truck] - 1)     #same_truck = 1 neu ca 2 item xep tren xe

                # i nam ben trai j 
                solver.Add(x[i] + w_i * (1 - o[i]) + h_i * o[i] <= x[j] + (1 - left[(i, j)]) * big_M)

                # i nam ben phai j
                solver.Add(x[j] + w_j * (1 - o[j]) + h_j * o[j] <= x[i] + (1 - right[(i, j)]) * big_M)

                # i nam duoi j
                solver.Add(y[i] + h_i * (1 - o[i]) + w_i * o[i] <= x[j] + (1 - below[(i, j)]) * big_M)

                # i nam tren j
                solver.Add(y[j] + h_j * (1 - o[j]) + w_j * o[j] <= x[i] + (1 - above[(i, j)]) * big_M)

                # Neu cung xe tai thi thoa man it nhat 1 trong 4 dieu kien
                solver.Add(left[(i, j)] + right[(i, j)] + below[(i, j)] + above[(i, j)] >= same_truck)

    objective = solver.Objective()
    for j in range(k):
        objective.SetCoefficient(u[j], cost[j])
    objective.SetMinimization()

    print("Solving...")
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print(f"Tổng chi phí tối ưu: {solver.Objective().Value()}")

        result = []
        for i in range(n):  
            truck_assigned = -1
            for j in range(k):
                if t[i][j].solution_value() > 0.5:
                    truck_assigned = j
                    break

            result.append((
                i + 1,                  # i
                truck_assigned + 1,     # t[i]
                int(x[i].solution_value()),
                int(y[i].solution_value()),
                int(o[i].solution_value())
            ))

            print(f"{i+1} {truck_assigned} {int(x[i].solution_value())} {int(y[i].solution_value())} {int(o[i].solution_value())}")

        return result

    else:
        print("Không tìm thấy giải pháp tối ưu")
        return None

def input():
    """
    Đọc dữ liệu đầu vào từ bàn phím và lưu ra file tạm 'input_temp.txt'
    """
    n = int(input("Nhập số lượng món hàng (n): "))
    k = int(input("Nhập số lượng xe tải (k): "))

    lines = []
    lines.append(f"{n} {k}")

    print("Nhập kích thước món hàng (width height), mỗi dòng 1 món:")
    for i in range(n):
        w, h = input(f"Món hàng {i+1}: ").split()
        lines.append(f"{w} {h}")

    print("Nhập kích thước và chi phí xe tải (width height cost), mỗi dòng 1 xe tải:")
    for j in range(k):
        w, h, c = input(f"Xe tải {j+1}: ").split()
        lines.append(f"{w} {h} {c}")

    # Ghi ra file tạm
    with open('input_temp.txt', 'w') as f:
        f.write('\n'.join(lines))

    return 'input_temp.txt'

    
def write_output(result, output_path):
    """
    Ghi kết quả vào file đầu ra theo định dạng yêu cầu
    """
    with open(output_path, 'w') as file:
        for item in result:
            i, t_i, x_i, y_i, o_i = item
            file.write(f"{i} {t_i} {x_i} {y_i} {o_i}\n")
    
def main():
    # Đường dẫn thư mục chứa test case và thư mục output
    testcase_folder = 'testcase'
    output_folder = 'output'

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)

    # Lặp qua tất cả file trong folder testcase
    for filename in os.listdir(testcase_folder):
        if filename.endswith('.txt'):
            input_file = os.path.join(testcase_folder, filename)
            output_file = os.path.join(output_folder, f'output_{filename}')

            print(f"\n🔧 Đang chạy test case: {filename}")
            result = solve(input_file)

            if result:
                write_output(result, output_file)
                n, k, data, _, _ = input(input_file)
            else:
                print(f"❌ Không tìm thấy lời giải cho {filename}")

if __name__ == "__main__":
    main()

