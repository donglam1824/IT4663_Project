import os

def read_input(filename):
    """Đọc dữ liệu đầu vào từ file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Đọc số lượng items và trucks
    N, K = map(int, lines[0].strip().split())
    
    # Đọc thông tin các items (width, length)
    items = []
    for i in range(1, N + 1):
        w, l = map(int, lines[i].strip().split())
        items.append((w, l, i))  # (width, length, item_id)
    
    # Đọc thông tin các trucks (width, length, cost)
    trucks = []
    for k in range(N + 1, N + 1 + K):
        W, L, c = map(int, lines[k].strip().split())
        trucks.append((W, L, c, k - N))  # (width, length, cost, truck_id)
    
    return N, K, items, trucks

class Shelf:
    """Lớp đại diện cho một shelf trong truck"""
    def __init__(self, y, height, truck_width):
        self.y = y  # Vị trí y của shelf
        self.height = height  # Chiều cao của shelf
        self.used_width = 0  # Chiều rộng đã sử dụng
        self.truck_width = truck_width  # Chiều rộng tối đa của truck
        self.items = []  # Danh sách items trên shelf này
    
    def can_fit(self, width, height):
        """Kiểm tra xem item có thể fit vào shelf không"""
        return (self.used_width + width <= self.truck_width and 
                height <= self.height)
    
    def add_item(self, item_id, width, height, rotated):
        """Thêm item vào shelf"""
        x = self.used_width
        y = self.y
        self.items.append((item_id, x, y, rotated))
        self.used_width += width
        return x, y

class Truck:
    """Lớp đại diện cho một truck"""
    def __init__(self, width, length, cost, truck_id):
        self.width = width
        self.length = length
        self.cost = cost
        self.truck_id = truck_id
        self.shelves = []  # Danh sách các shelf
        self.used_height = 0  # Chiều cao đã sử dụng
        self.items = []  # Danh sách tất cả items trong truck
    
    def can_fit_item(self, item_width, item_length):
        """Kiểm tra xem item có thể fit vào truck không (có thể xoay)"""
        # Thử không xoay
        if self.can_fit_without_rotation(item_width, item_length):
            return True, False
        # Thử xoay 90 độ
        if self.can_fit_without_rotation(item_length, item_width):
            return True, True
        return False, False
    
    def can_fit_without_rotation(self, width, height):
        """Kiểm tra fit mà không xoay"""
        # Thử fit vào các shelf hiện có
        for shelf in self.shelves:
            if shelf.can_fit(width, height):
                return True
        
        # Thử tạo shelf mới
        if self.used_height + height <= self.length:
            return True
        
        return False
    
    def add_item(self, item_id, item_width, item_length, rotated=False):
        """Thêm item vào truck"""
        if rotated:
            width, height = item_length, item_width
        else:
            width, height = item_width, item_length
        
        # Thử thêm vào shelf hiện có
        for shelf in self.shelves:
            if shelf.can_fit(width, height):
                x, y = shelf.add_item(item_id, width, height, 1 if rotated else 0)
                self.items.append((item_id, x, y, 1 if rotated else 0))
                return True
        
        # Tạo shelf mới nếu có thể
        if self.used_height + height <= self.length:
            new_shelf = Shelf(self.used_height, height, self.width)
            x, y = new_shelf.add_item(item_id, width, height, 1 if rotated else 0)
            self.shelves.append(new_shelf)
            self.used_height += height
            self.items.append((item_id, x, y, 1 if rotated else 0))
            return True
        
        return False

def solve_container_loading(N, K, items, trucks):
    """Giải bài toán Container 2D Loading bằng Greedy + Shelf"""
    
    # Sắp xếp trucks theo chi phí tăng dần (greedy: chọn truck rẻ nhất trước)
    trucks.sort(key=lambda x: x[2])
    
    # Sắp xếp items theo diện tích giảm dần (greedy: xếp item lớn trước)
    items.sort(key=lambda x: x[0] * x[1], reverse=True)
    
    # Tạo các truck objects
    truck_objects = []
    for W, L, c, truck_id in trucks:
        truck_objects.append(Truck(W, L, c, truck_id))
    
    # Kết quả lưu trữ
    solution = {}
    used_trucks = set()
    
    # Greedy: Thử xếp từng item vào truck có chi phí thấp nhất có thể
    for item_width, item_length, item_id in items:
        placed = False
        
        # Thử xếp vào các truck đã sử dụng trước (để tối ưu chi phí)
        for truck in truck_objects:
            if truck.truck_id in used_trucks:
                can_fit, need_rotation = truck.can_fit_item(item_width, item_length)
                if can_fit:
                    success = truck.add_item(item_id, item_width, item_length, need_rotation)
                    if success:
                        placed = True
                        break
        
        # Nếu chưa xếp được, thử truck mới (theo thứ tự chi phí tăng dần)
        if not placed:
            for truck in truck_objects:
                if truck.truck_id not in used_trucks:
                    can_fit, need_rotation = truck.can_fit_item(item_width, item_length)
                    if can_fit:
                        success = truck.add_item(item_id, item_width, item_length, need_rotation)
                        if success:
                            used_trucks.add(truck.truck_id)
                            placed = True
                            break
        
        if not placed:
            print(f"Không thể xếp item {item_id}")
            return None, None
    
    # Tổng hợp kết quả
    for truck in truck_objects:
        if truck.truck_id in used_trucks:
            for item_id, x, y, rotated in truck.items:
                solution[item_id] = (truck.truck_id, x, y, rotated)
    
    # Tính tổng chi phí
    total_cost = 0
    for truck in truck_objects:
        if truck.truck_id in used_trucks:
            total_cost += truck.cost
    
    return solution, total_cost

def main():
    """Hàm chính"""
    # Đọc dữ liệu từ file
    filename = "Code/testcase/test10.txt"  # Tên file input
    N, K, items, trucks = read_input(filename)
    
    # Giải bài toán
    solution, total_cost = solve_container_loading(N, K, items, trucks)
    
    if solution:
        # Tạo output lines
        output_lines = []
        
        for i in range(1, N + 1):
            if i in solution:
                truck_id, x, y, rotated = solution[i]
                output_lines.append(f"{i} {truck_id} {x} {y} {rotated}")
            else:
                output_lines.append(f"{i} -1 -1 -1 -1")
        
        # Thêm tổng chi phí
        output_lines.append(f"Total cost: {total_cost}")
        
        # In kết quả ra console
        for line in output_lines:
            print(line)
        
        # Tạo thư mục Result nếu chưa có
        result_dir = "Code/Greedy_Shelf/Result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # Xuất kết quả ra file
        output_file = os.path.join(result_dir, "output10.txt")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in output_lines:
                    f.write(line + '\n')
            print(f"\nKết quả đã được xuất ra file: {output_file}")
        except Exception as e:
            print(f"Lỗi khi xuất file: {e}")
    else:
        print("Không tìm được lời giải")

if __name__ == "__main__":
    main()